import json
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import argparse
import numpy as np
from tqdm import tqdm


def preprocess_data(data):
    question_with_knowledge = []
    for d in data:
        question = d['query']
        for k in d['knowledges']:
            question_with_knowledge.append({'query': question, 'cands': d['cands'], 'answer': d['answer'], 'knowledge': k})
    return question_with_knowledge


def batching_list_instances(args, data):
    batch_size = args.batch_size
    num_data = len(data)
    total_batch = num_data // batch_size
    batch_data = []
    for batch_id in range(total_batch):
        one_batch_data = data[batch_id * batch_size:(batch_id + 1) * batch_size]
        batch_data.append(one_batch_data)
    return batch_data


def simple_batching(args, batch_data, tokenizer, word_pad_idx=0):
    inputs_ids = []
    attention_masks = []
    max_inputs_len = 0
    labels_ids = []
    max_labels_len = 0

    for idx, data in enumerate(batch_data):
        input_tokens = tokenizer([f"{data['knowledge']} {data['query']}"] * len(data['cands']))
        input_ids = input_tokens.input_ids
        attention_mask = input_tokens.attention_mask
        max_inputs_len = max(max_inputs_len, len(input_ids[0]))
        label_tokens = tokenizer([f"<extra_id_0> {cand} <extra_id_1>" for cand in data['cands']], padding='longest')
        label_ids = label_tokens.input_ids
        max_labels_len = max(max_labels_len, len(label_ids[0]))
        inputs_ids.extend(input_ids)
        attention_masks.extend(attention_mask)
        labels_ids.extend(label_ids)

    for i, input_ids in enumerate(inputs_ids):
        pad_word_num = max_inputs_len - len(input_ids)
        inputs_ids[i].extend([word_pad_idx] * pad_word_num)
        attention_masks[i].extend([word_pad_idx] * pad_word_num)

    for i, label_ids in enumerate(labels_ids):
        pad_word_num = max_labels_len - len(label_ids)
        labels_ids[i].extend([word_pad_idx] * pad_word_num)

    inputs_ids = torch.LongTensor(inputs_ids).to(args.device)
    attention_masks = torch.LongTensor(attention_masks).to(args.device)
    labels_ids = torch.LongTensor(labels_ids).to(args.device)
    labels_ids[labels_ids == 0] = -100
    return inputs_ids, attention_masks, labels_ids


def score_for_input(args, tokenizer, model, raw_data):
    batch_data = batching_list_instances(args, raw_data)
    print("num of batches: ", len(batch_data))
    num_cands = len(raw_data[0]['cands'])
    tbar = tqdm(batch_data)
    correct, total = 0, 0
    for data in tbar:
        inputs_ids, attention_masks, labels_ids = simple_batching(args, data, tokenizer)
        # print(inputs_ids)
        # print(attention_masks)
        # print(labels_ids)
        with torch.no_grad():
            logits = model(input_ids=inputs_ids, attention_mask=attention_masks, labels=labels_ids).logits
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        # 在这里不使用average-loss 则需要吧loss的item数值乘以labels的有效长度
        scores = torch.Tensor([-loss_fct(logits[i], labels_ids[i]).item() * sum(labels_ids[i] != -100).item() for i in range(len(inputs_ids))])
        scores = scores.reshape(-1, num_cands)  # [batch_size, num_cands]
        probs = torch.softmax(scores, dim=1)
        for i in range(len(data)):
            data[i]['prob'] = probs[i].tolist()
            data[i]['pred'] = probs[i].argmax().item()
            data[i]['ok'] = 1 if data[i]['cands'][data[i]['pred']] == data[i]['answer'] else 0
            correct += data[i]['ok']
            total += 1
        tbar.set_postfix({'accuracy: ': correct / total})
    print("num of data with true knowledge: ", correct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'])
    parser.add_argument('--task', type=str, default='csqa', choices=['csqa', 'csqa2', 'numersense', 'qasc'])
    parser.add_argument('--model_name', type=str, default='pretrained_models/t5-3b')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    args.output_path = f'data/{args.task}/filter_results/{args.model_name.split("/")[-1]}_filtered_{args.input_path.split("/")[-1]}'
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir="pretrained_models/")
    model.to(args.device)
    with open(args.input_path) as f:
        jsdata = json.load(f)
    raw_data = preprocess_data(jsdata)
    print("num of data: ", len(raw_data))
    score_for_input(args, tokenizer, model, raw_data)
    raw_data = raw_data[:len(raw_data) // args.batch_size * args.batch_size]
    print("num of result_data:", len(raw_data))
    with open(args.output_path, "w") as f:
        json.dump(raw_data, f, indent=4)


if __name__ == '__main__':
    main()
