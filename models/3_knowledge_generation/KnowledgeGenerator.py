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
        self.vocab_size = self.generator.config.vocab_size
        self.ignore_index = args.ignore_index
        self.loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
        self.linear_layer = nn.Sequential(
                                nn.Linear(self.hidden_size, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, self.hidden_size)
                                          )
        self.cos = CosineSimilarity(dim=-1)
        self.without_contrastive = args.without_contrastive
        self.alpha = args.alpha

    def affine_transformation(self, input_features, padding_mask, axis=1):
        length = torch.sum(padding_mask, dim=1) - 1
        padding_mask = seq_len_to_mask(length, max_len=padding_mask.shape[-1])
        trans_tmp = F.relu(self.linear_layer(input_features))  # batch
        trans_tmp = trans_tmp * padding_mask.unsqueeze(-1).float()
        trans_emb = torch.sum(trans_tmp, dim=axis) / length.unsqueeze(-1)
        return trans_emb

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, args):
        self.generator.eval()
        ret_dict = self.generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=8,
            num_beams=10,
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
        if self.without_contrastive:
            return cand_ids[0]
        cand_mask = (cand_ids != self.pad_id).long()
        cand_len = torch.sum(cand_mask, dim=-1)
        max_len = torch.max(cand_len).item()
        cand_ids = cand_ids[:, :max_len]
        beam_indices = ret_dict['beam_indices']
        # [num_return_sequences, seq_len+1]
        beam_indices = torch.where(beam_indices > 0, beam_indices, 0)
        decoder_hidden_states = ret_dict["decoder_hidden_states"]
        # get the hidden_states from the last layer of decoder
        hidden_states_from_output = torch.cat([decoder_hidden_states[i][-1] for i in range(len(decoder_hidden_states))], dim=1)  # beam, seq_len, h
        h = hidden_states_from_output.shape[-1]
        # dim=0
        # beam_indices -> [num_return_sequences, seq_len+1, h]
        decoder_hidden_states = torch.gather(hidden_states_from_output, 0, beam_indices[:, :-1].unsqueeze(-1).repeat(1, 1, h))
        encoder_hidden_states = ret_dict["encoder_hidden_states"][-1]  # [src_len, h]
        encoder_feature = self.affine_transformation(encoder_hidden_states, attention_mask)  # [1, h]
        decoder_feature = self.affine_transformation(decoder_hidden_states, cand_mask[:, :-1])  # [beam_size, h]
        cos_distance = self.cos(encoder_feature, decoder_feature)  # beam_size
        cos_distance = cos_distance.exp()
        index = torch.argmax(cos_distance)  # beam_size
        result = cand_ids[index]
        return result  # [sequence_length]

    def forward(self, query_ids, knowledge_input_ids, knowledge_output_ids, p_n_tag):
        encoder = self.generator.get_encoder()
        decoder = self.generator.get_decoder()

        batch_size = query_ids.size(0)
        query_attention_mask = ~(query_ids == self.pad_id)
        encoder_hidden_states = encoder(query_ids, query_attention_mask)['last_hidden_state']
        knowledge_input_attention_mask = ~(knowledge_input_ids == self.pad_id)
        knowledge_input_attention_mask[:, 0] = 1
        decoder_out = decoder(input_ids=knowledge_input_ids, attention_mask=knowledge_input_attention_mask,
                              encoder_hidden_states=encoder_hidden_states,
                              encoder_attention_mask=query_attention_mask)  # last layer
        if "t5" in self.LM:
            decoder_last_layer = decoder_out[0] * (self.generator.model_dim ** -0.5)
        else:
            decoder_last_layer = decoder_out[0]
        lm_logits = self.generator.lm_head(decoder_last_layer)
        # negative log likelihood (NLL) loss
        knowledge_output_ids[knowledge_output_ids == 0] = self.ignore_index
        if self.without_contrastive:
            nll_loss = self.loss_fct(lm_logits.permute(0, 2, 1), knowledge_output_ids)
            return {'nll_loss': nll_loss}
        positive_mask = (p_n_tag == 1)
        nll_loss = self.loss_fct(lm_logits[positive_mask].permute(0, 2, 1), knowledge_output_ids[positive_mask])
        # nll_loss = self.loss_fct(lm_logits.permute(0, 2, 1), knowledge_output_ids)

        # negative sample
        if sum(p_n_tag) != batch_size:
            negative_mask = (p_n_tag == 0)
            pos_num = sum(p_n_tag)
            neg_num = batch_size - pos_num
            positive_encoder_hidden_states = self.affine_transformation(
                encoder_hidden_states[positive_mask], query_attention_mask[positive_mask])
            positive_decoder_hidden_states = self.affine_transformation(
                decoder_last_layer[positive_mask], knowledge_input_attention_mask[positive_mask])
            negative_decoder_hidden_states = self.affine_transformation(
                decoder_last_layer[negative_mask], knowledge_input_attention_mask[negative_mask])
            pos_pos_examples_cos = self.cos(positive_encoder_hidden_states, positive_decoder_hidden_states)
            pos_pos_examples_cos = pos_pos_examples_cos.exp()
            pos_neg_examples_cos = self.cos(
                positive_encoder_hidden_states.repeat_interleave(neg_num, dim=0),  # [neg_num*pos_num, h]
                negative_decoder_hidden_states.repeat(pos_num, 1)  # [neg_num*pos_num, h]
            )  # [neg_num*pos_num]
            pos_neg_examples_cos = pos_neg_examples_cos.exp()
            pos_neg_examples_cos = pos_neg_examples_cos.reshape(neg_num, pos_num).sum(0)

            # pos_pos_examples_cos = pos_pos_examples_cos.sum()
            # pos_neg_examples_cos = pos_neg_examples_cos.sum()

            examples_cos = pos_pos_examples_cos / pos_neg_examples_cos
            nce_loss = torch.sum(-torch.log(examples_cos))
            return {'total_loss': nll_loss + self.alpha * nce_loss, 'nll_loss': nll_loss,
                    'nce_loss': nce_loss}
        else:
            return {'total_loss': nll_loss, 'nll_loss': nll_loss}
