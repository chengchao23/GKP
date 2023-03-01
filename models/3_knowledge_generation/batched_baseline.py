import sys
import argparse
import os
import numpy as np
import json
import random
import time
from termcolor import colored
from tqdm import tqdm

import torch
from torch import optim
import torch.cuda

from transformers import T5ForConditionalGeneration, AutoTokenizer, get_linear_schedule_with_warmup


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def batching_list_instances(batch_size, data, shffule=True):
    """
    List of data -> List of batched_data
    """
    if shffule:
        data.sort(key=lambda x: len(x['query']))
    train_num = len(data)
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_data = data[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(one_batch_data)
    if shffule:
        random.shuffle(batched_data)
    print("num of batches: ", len(batched_data))
    return batched_data


def simple_batching(batch_data, tokenizer, word_pad_idx, args):
    """
    batching these instances together and return tensors. The seq_tensors for word and char contain their word id and char id.
    :return
        sent_seq_len: Shape: (batch_size), the length of each paragraph in a batch.
        sent_tensor: Shape: (batch_size, max_seq_len, max_token_num)
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    query_ids = []
    max_query_len = 0
    knowledge_ids = []
    max_knowledge_len = 0

    input_ids = []
    input_attention_mask = []
    max_input_len = 0

    for idx, data in enumerate(batch_data):
        if 't5' in args.LM:
            query = tokenizer(f"{args.task_prefix} {data['query']} {', '.join(data['cands'])} explanation:")
            knowledge = tokenizer(data['knowledge'])
            query_idx = query['input_ids']
            knowledge_idx = knowledge['input_ids']
            max_query_len = max(max_query_len, len(query_idx))
            max_knowledge_len = max(max_knowledge_len, len(knowledge_idx))
            query_ids.append(query_idx)
            knowledge_ids.append(knowledge_idx)
        elif 'gpt' in args.LM:
            input = tokenizer(f"{tokenizer.bos_token} {data['query']} {data['knowledge']} {tokenizer.eos_token}")
            input_idx, input_mask = input['input_ids'], input['attention_mask']
            max_input_len = max(max_input_len, len(input_idx))
            input_ids.append(input_idx)
            input_attention_mask.append(input_mask)
        else:
            raise NotImplementedError(f"args.LM {args.LM}is not implemented")

    # padding: batch_size, max_query_len
    if 't5' in args.LM:
        for i, query_idx in enumerate(query_ids):
            pad_word_num = max_query_len - len(query_idx)
            query_ids[i].extend([word_pad_idx] * pad_word_num)
        for i, knowledge_idx in enumerate(knowledge_ids):
            pad_word_num = max_knowledge_len - len(knowledge_idx)
            knowledge_ids[i].extend([word_pad_idx] * pad_word_num)

        query_ids = torch.LongTensor(query_ids).to(args.device)
        knowledge_ids = torch.LongTensor(knowledge_ids).to(args.device)
        knowledge_ids[knowledge_ids == 0] = -100
        return query_ids, knowledge_ids
    elif 'gpt' in args.LM:
        for i, input_idx in enumerate(input_ids):
            pad_word_num = max_input_len - len(input_idx)
            input_ids[i].extend([word_pad_idx] * pad_word_num)
            input_attention_mask[i].extend([0] * pad_word_num)
        input_ids = torch.LongTensor(input_ids).to(args.device)
        input_attention_mask = torch.LongTensor(input_attention_mask).to(args.device)
        return input_ids, input_attention_mask
    else:
        raise RuntimeError("args.LM is not implemented")


def train(model, tokenizer, train_data, dev_data, warmup_steps, args):
    epochs = args.num_epochs
    lr = args.lr
    model = model.to(args.device)
    # Count the number of parameters.
    num_param = 0
    for idx in list(model.parameters()):
        try:
            num_param += idx.size()[0] * idx.size()[1]
        except IndexError:
            num_param += idx.size()[0]
    print('num_param: ', num_param)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif args.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        raise RuntimeError("args.optimizer is not implemented")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)
    lowest_loss = 0

    batched_data = batching_list_instances(batch_size, train_data)
    for i in tqdm(range(1, epochs + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.train()
        model.zero_grad()
        for index in tqdm(np.random.permutation(len(batched_data))):
            query_ids, knowledge_ids = simple_batching(batched_data[index], tokenizer, 0, args)
            attention_mask = ~(query_ids == 0)
            loss = model(input_ids=query_ids, attention_mask=attention_mask, labels=knowledge_ids).loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        end_time = time.time()
        print("Epoch %d -> Loss: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)
        model.eval()
        if i == 1:
            lowest_loss = epoch_loss
            evaluate_model(model, dev_data, 'dev', tokenizer, args)
        else:
            if epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                evaluate_model(model, dev_data, 'dev', tokenizer, args)
            else:
                print(colored('epoch_loss > lowest_loss', 'red'))


def evaluate_model(model, data, name, tokenizer, args):
    print(colored('[Begin evaluating ' + name + '.]', 'green'))
    # generate by sample.
    with torch.no_grad():
        for d in tqdm(data):
            query = f"{args.task_prefix} {d['query']} {', '.join(d['cands'])} explanation:"
            query = tokenizer(query, return_tensors='pt').to(args.device)
            knowledge_ids = model.generate(input_ids=query['input_ids'], attention_mask=query['attention_mask'])
            knowledge = tokenizer.decode(knowledge_ids, skip_special_tokens=True)
            d['knowledge'] = knowledge
    result_file = os.path.join(args.result_dir,
                               f"{args.number}_{args.train_data_file.split('/')[-1].split('.')[0]}_{args.LM.split('/')[-1]}_{args.dev_data_file.split('/')[-1]}")
    print(colored('[The result will writing to' + result_file + '.]', 'blue'))
    with open(result_file, 'w') as f:
        json.dump(data, f, indent=4)


def parse_arguments(parser):
    # Training hyper-parameters
    parser.add_argument('--number', type=str, required=True, default=0)
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--train_data_file', type=str,
                        default="data/csqa/filter_results/")
    parser.add_argument('--dev_data_file', type=str,
                        default="data/csqa/dev.csqa.json")
    parser.add_argument('--result_dir', type=str, default="data/csqa/baseline_results")
    # parser.add_argument('--model_dir', type=str, default="")
    # parser.add_argument('--model_folder', type=str, default="", help="The name to save the model files")
    parser.add_argument('--batch_size', type=int, default=12, help="default batch size is 12")
    parser.add_argument('--num_epochs', type=int, default=7, help="Usually we set to 7.")

    # model hyperparameter
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--LM', type=str, default='pretrained_models/t5-large')
    parser.add_argument('--task_prefix', type=str, default="Generating explanations for question with candidates:")
    parser.add_argument('--only_oks', action='store_true')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Prompt knowledge for QA")
    args = parse_arguments(parser)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.LM)
    model = T5ForConditionalGeneration.from_pretrained(args.LM)

    train_data = []
    with open(args.train_data_file) as f:
        jsdata = json.load(f)
    print('num of jsdata:', len(jsdata))
    for data in jsdata:
        # 去除样本中的错误知识
        if args.only_oks:
            if data['ok']:
                train_data.append(data)
        else:
            train_data.append(data)
    print('num of train data:', len(train_data))
    with open(args.dev_data_file) as f:
        dev_data = json.load(f)
    print('num of dev data:', len(dev_data))
    train(model, tokenizer, train_data, dev_data, 1000, args)


if __name__ == "__main__":
    main()
