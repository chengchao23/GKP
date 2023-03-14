import sys

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, generation_utils
import torch.nn.functional as F
from torch.nn import CosineSimilarity


def seq_len_to_mask(seq_len, max_len):
    batch_size = seq_len.shape[0]
    broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
    mask = broad_cast_seq_len < seq_len.unsqueeze(1)
    return mask


class KnowledgeGenerator(nn.Module):
    def __init__(self, args):
        super(KnowledgeGenerator, self).__init__()
        if 't5' in args.LM:
            self.generator = T5ForConditionalGeneration.from_pretrained(args.LM)
        else:
            raise NotImplementedError(f"not support {args.LM} yet")
        self.LM = args.LM
        self.pad_id = args.pad_id
        self.hidden_size = self.generator.config.hidden_size
        self.ignore_index = args.ignore_index
        self.loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
        self.query_linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.knowledge_linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        # self.cross_attention = nn.MultiheadAttention(num_heads=2, embed_dim=self.hidden_size, dropout=0.5)
        self.cos = CosineSimilarity(dim=-1)
        self.loss_func = args.loss_func

    def query_affine_transformation(self, input_features, padding_mask, axis=1):
        length = torch.sum(padding_mask, dim=1) - 1
        padding_mask = seq_len_to_mask(length, max_len=padding_mask.shape[-1])
        trans_tmp = F.relu(self.query_linear_layer(input_features))  # batch
        trans_tmp = trans_tmp * padding_mask.unsqueeze(-1).float()
        trans_emb = torch.sum(trans_tmp, dim=axis) / length.unsqueeze(-1)
        return trans_emb

    def knowledge_affine_transformation(self, input_features, padding_mask, axis=1):
        length = torch.sum(padding_mask, dim=1) - 1
        padding_mask = seq_len_to_mask(length, max_len=padding_mask.shape[-1])
        trans_tmp = F.relu(self.knowledge_linear_layer(input_features))  # batch
        trans_tmp = trans_tmp * padding_mask.unsqueeze(-1).float()
        trans_emb = torch.sum(trans_tmp, dim=axis) / length.unsqueeze(-1)
        return trans_emb

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, args):
        self.generator.eval()
        ret_dict = self.generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            num_return_sequences=args.beam_size,
            num_beams=args.beam_size,
            max_length=args.max_length + 2,
            # +2 from original because we start at step=1 and stop before max_length
            min_length=args.min_length + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=args.no_repeat_ngram,
            length_penalty=args.length_pen,
            early_stopping=args.early_stop,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        cand_ids = ret_dict["sequences"]  # [beam_size, seq_len]
        sequences_scores = ret_dict["sequences_scores"]  # beam_size
        normalize = torch.sum(sequences_scores, keepdim=True, dim=-1)
        cand_mask = (cand_ids != self.pad_id).long()
        cand_len = torch.sum(cand_mask, dim=-1)
        max_len = torch.max(cand_len).item()
        cand_ids = cand_ids[:, :max_len]
        beam_indices = ret_dict['beam_indices']
        # [num_return_sequences, seq_len+1]
        beam_indices = torch.where(beam_indices > 0, beam_indices, 0)
        decoder_hidden_states = ret_dict["decoder_hidden_states"]
        # get the hidden_states from the last layer of decoder
        hidden_states_from_output = torch.cat([decoder_hidden_states[i][-1] for i in range(len(decoder_hidden_states))],
                                              dim=1)  # beam, seq_len, h
        h = hidden_states_from_output.shape[-1]
        # dim=0
        # beam_indices -> [num_return_sequences, seq_len+1, h]
        decoder_hidden_states = torch.gather(hidden_states_from_output, 0,
                                             beam_indices[:, :-1].unsqueeze(-1).repeat(1, 1, h))
        encoder_hidden_states = ret_dict["encoder_hidden_states"][-1]  # [src_len, h]
        encoder_feature = self.query_affine_transformation(encoder_hidden_states, attention_mask)  # [1, h]
        decoder_feature = self.knowledge_affine_transformation(decoder_hidden_states,
                                                               cand_mask[:, :-1])  # [beam_size, h]
        cos_distance = self.cos(encoder_feature, decoder_feature)  # beam_size

        scores = (sequences_scores / normalize) + cos_distance
        max_index = torch.argmax(scores, dim=-1)  # 1
        result = cand_ids[max_index]
        return result  # [sequence_length]

    def forward(self, query_ids, knowledge_input_ids, knowledge_output_ids, p_n_tag):
        encoder = self.generator.get_encoder()
        decoder = self.generator.get_decoder()

        num_samples = query_ids.size(0)
        positive_mask = (p_n_tag == 1)
        pos_num = sum(p_n_tag)
        # print(query_ids.shape)
        # print(p_n_tag)
        positive_query_ids = query_ids[positive_mask]  # [pos_num, query_len]
        positive_query_attention_mask = ~(positive_query_ids == self.pad_id)
        positive_encoder_hidden_states = encoder(positive_query_ids, positive_query_attention_mask)[
            'last_hidden_state']  # [pos_num, query_len, hidden_size]
        positive_knowledge_input_ids = knowledge_input_ids[positive_mask]  # [pos_num, knowledge_len]
        positive_knowledge_output_ids = knowledge_output_ids[positive_mask]  # [pos_num, knowledge_len]
        positive_knowledge_input_attention_mask = ~(positive_knowledge_input_ids == self.pad_id)
        positive_knowledge_input_attention_mask[:, 0] = 1  # [pos_num, knowledge_len]
        positive_decoder_out = decoder(input_ids=positive_knowledge_input_ids,
                                       attention_mask=positive_knowledge_input_attention_mask,
                                       encoder_hidden_states=positive_encoder_hidden_states,
                                       encoder_attention_mask=positive_query_attention_mask)  # last layer
        positive_decoder_last_layer = positive_decoder_out[0] * (self.generator.model_dim ** -0.5)
        # [pos_num, knowledge_len, hidden_size]

        positive_lm_logits = self.generator.lm_head(positive_decoder_last_layer)
        # negative log likelihood (NLL) loss for positive samples
        positive_knowledge_output_ids[positive_knowledge_output_ids == 0] = self.ignore_index
        positive_nll_loss = self.loss_fct(positive_lm_logits.permute(0, 2, 1), positive_knowledge_output_ids)
        if sum(p_n_tag) == num_samples:
            return {'total_loss': positive_nll_loss, 'nll_loss': positive_nll_loss}

        # Contrastive learning
        negative_mask = (p_n_tag == 0)
        neg_num = num_samples - pos_num
        negative_query_ids = query_ids[negative_mask]
        negative_query_attention_mask = ~(negative_query_ids == self.pad_id)
        negative_encoder_hidden_states = encoder(negative_query_ids, negative_query_attention_mask)[
            'last_hidden_state']  # [neg_num, query_len, hidden_size]
        negative_knowledge_input_ids = knowledge_input_ids[negative_mask]  # [pos_num, knowledge_len]
        negative_knowledge_input_attention_mask = ~(negative_knowledge_input_ids == self.pad_id)
        negative_knowledge_input_attention_mask[:, 0] = 1  # [pos_num, knowledge]
        negative_decoder_out = decoder(input_ids=negative_knowledge_input_ids,
                                       attention_mask=negative_knowledge_input_attention_mask,
                                       encoder_hidden_states=negative_encoder_hidden_states,
                                       encoder_attention_mask=negative_query_attention_mask)
        negative_decoder_last_layer = negative_decoder_out[0] * (self.generator.model_dim ** -0.5)
        # [0] -> last layer
        # [neg_num, knowledge_len, hidden_size]

        positive_encoder_hidden_states = self.query_affine_transformation(
            positive_encoder_hidden_states, positive_query_attention_mask)

        positive_decoder_hidden_states = self.knowledge_affine_transformation(
            positive_decoder_last_layer, positive_knowledge_input_attention_mask)

        negative_decoder_hidden_states = self.knowledge_affine_transformation(
            negative_decoder_last_layer, negative_knowledge_input_attention_mask)

        pos_pos_examples_cos = self.cos(positive_encoder_hidden_states, positive_decoder_hidden_states)

        pos_neg_examples_cos = self.cos(
            positive_encoder_hidden_states.repeat_interleave(neg_num, dim=0),  # [pos_num * neg_num, h]
            negative_decoder_hidden_states.repeat(pos_num, 1)  # [pos_num * neg_num, h]
        )  # [neg_num*pos_num]

        if self.loss_func == 'nce_loss':
            pos_neg_examples_cos = pos_neg_examples_cos.reshape(pos_num, neg_num).sum(-1)
            examples_cos = pos_pos_examples_cos / pos_neg_examples_cos
            nce_loss = torch.mean(-torch.log(examples_cos))
            return {'total_loss': positive_nll_loss + nce_loss, 'nll_loss': positive_nll_loss,
                    'nce_loss': nce_loss}
        elif self.loss_func == 'pair_loss':
            pos_pos_examples_cos = pos_pos_examples_cos.repeat_interleave(neg_num, dim=0)  # [neg_num*pos_num]
            ones = torch.ones(pos_pos_examples_cos.size(), device=pos_pos_examples_cos.device)
            marginLoss = nn.MarginRankingLoss(reduction='sum')
            pair_loss = marginLoss(pos_pos_examples_cos, pos_neg_examples_cos, ones)
            return {'total_loss': positive_nll_loss + pair_loss, 'nll_loss': positive_nll_loss,
                    'pair_loss': pair_loss}
        else:
            raise NotImplementedError(f'{args.loss_func} is not implemented')
