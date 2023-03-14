import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import argparse
import numpy as np
from tqdm import tqdm


def preprocess_data(data):
    question_with_knowledge = []
    for i, d in enumerate(data):
        question = d['query']
        for k in d['knowledges']:
            if len(k) != 0:
                question_with_knowledge.append(
                    {'query': question, 'cands': d['cands'], 'answer': d['answer'], 'knowledge': k})
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


def simple_batching(args, batch_data, tokenizer, max_length=512, word_pad_idx=0):
    inputs_ids = []
    inputs_token_type_ids = []
    attention_masks = []
    max_inputs_len = 0

    for idx, data in enumerate(batch_data):
        cands = ", ".join(data['cands'])
        tokenized_input_seq_pair = tokenizer.encode_plus(f"{data['query']} {cands}", data['knowledge'],
                                                         max_length=max_length, return_token_type_ids=True,
                                                         truncation=True)
        input_ids = tokenized_input_seq_pair['input_ids']
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        if 'bart' not in args.model_name:
            token_type_ids = tokenized_input_seq_pair['token_type_ids']
        attention_mask = tokenized_input_seq_pair['attention_mask']
        inputs_ids.append(input_ids)
        if 'bart' not in args.model_name:
            inputs_token_type_ids.append(token_type_ids)
        attention_masks.append(attention_mask)
        max_inputs_len = max(max_inputs_len, len(input_ids))
    for i, input_ids in enumerate(inputs_ids):
        pad_word_num = max_inputs_len - len(input_ids)
        inputs_ids[i].extend([word_pad_idx] * pad_word_num)
        if 'bart' not in args.model_name:
            inputs_token_type_ids[i].extend([word_pad_idx] * pad_word_num)
        attention_masks[i].extend([word_pad_idx] * pad_word_num)

    inputs_ids = torch.LongTensor(inputs_ids).to(args.device)
    if 'bart' not in args.model_name:
        inputs_token_type_ids = torch.LongTensor(inputs_token_type_ids).to(args.device)
    attention_masks = torch.LongTensor(attention_masks).to(args.device)
    if 'bart' not in args.model_name:
        return inputs_ids, inputs_token_type_ids, attention_masks
    else:
        return inputs_ids, attention_masks


def score_for_input(args, model, tokenizer, raw_data):
    batch_data = batching_list_instances(args, raw_data)
    print("num of batches: ", len(batch_data))
    correct, total = 0, 0
    tbar = tqdm(batch_data)
    samples_data = []
    temp_sample = {'query': batch_data[0][0]['query'],
                   'cands': batch_data[0][0]['cands'],
                   'answer': batch_data[0][0]['answer'],
                   'knowledges': [],
                   'scores': [],
                   'oks': []}

    for data in tbar:
        if 'bart' not in args.model_name:
            inputs_ids, inputs_token_type_ids, attention_masks = simple_batching(args, data, tokenizer)
            with torch.no_grad():
                outputs = model(inputs_ids, attention_mask=attention_masks,
                                token_type_ids=inputs_token_type_ids, labels=None, return_dict=True)
        else:
            inputs_ids, attention_masks = simple_batching(args, data, tokenizer)
            with torch.no_grad():
                outputs = model(inputs_ids, attention_mask=attention_masks, labels=None, return_dict=True)
        scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()  # [batch_size, 3]

        for i in range(len(data)):
            data[i]['score'] = scores[i].tolist()
            # 筛选规则1：问题和知识的关系为蕴含关系
            if args.nli_rational == 'entail':
                data[i]['ok'] = 1 if np.argmax(scores[i]) == 0 else 0
            # 筛选规则2：问题和知识的关系为蕴含关系或中立关系（即非对立关系）
            elif args.nli_rational == 'entail_neutral':
                data[i]['ok'] = 1 if np.argmax(scores[i]) != 2 else 0
            else:
                raise RuntimeError("args.nli_rational didn't has choice:", {args.nli_rational})
            correct += data[i]['ok']
            total += 1
            if data[i]['query'] != temp_sample['query']:
                # print(temp_sample)
                samples_data.append(temp_sample)
                temp_sample = {'query': data[i]['query'],
                               'cands': data[i]['cands'],
                               'answer': data[i]['answer'],
                               'knowledges': [],
                               'scores': [],
                               'oks': []}
            temp_sample['knowledges'].append(data[i]['knowledge'])
            temp_sample['scores'].append(data[i]['score'])
            temp_sample['oks'].append(data[i]['ok'])

        tbar.set_postfix({'Accuracy: ': correct / total})
    samples_data.append(temp_sample)
    print("num of data with true knowledge: ", correct)
    return samples_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='csqa', choices=['csqa', 'qasc', 'csqa2'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'])
    parser.add_argument('--batch_size', type=int, default=8)  # in this case, it is better to set 8 to batch_size
    parser.add_argument('--model_name', type=str, default='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--nli_rational', type=str, default='entail', choices=['entail', 'entail_neutral'])
    args = parser.parse_args()
    args.output1_path = f'data/{args.task}/filter_results/{args.model_name.split("/")[-1]}_filtered_batched_{args.nli_rational}_{args.input_path.split("/")[-1]}'
    args.output2_path = f'data/{args.task}/filter_results/{args.model_name.split("/")[-1]}_filtered_sampled_{args.nli_rational}_{args.input_path.split("/")[-1]}'
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, cache_dir="pretrained_models/")
    model.to(args.device)
    with open(args.input_path) as f:
        jsdata = json.load(f)
    print("num of jsdata: ", len(jsdata))
    raw_data = preprocess_data(jsdata)
    print("num of single data:", len(raw_data))
    samples_data = score_for_input(args, model, tokenizer, raw_data)
    print('num of samples_data: ', len(samples_data))
    raw_data = raw_data[:len(raw_data) // args.batch_size * args.batch_size]
    print("num of result_data:", len(raw_data))
    with open(args.output1_path, "w") as f:
        json.dump(raw_data, f, indent=4)
    with open(args.output2_path, "w") as f:
        json.dump(samples_data, f, indent=4)


if __name__ == '__main__':
    main()
