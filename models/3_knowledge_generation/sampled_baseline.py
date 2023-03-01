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


def simple_batching_samples(data, tokenizer, args):
    query = tokenizer(f"{args.task_prefix} {data['query']} {','.join(data['cands'])} explanation:")
    query_ids = query['input_ids']
    query_ids = torch.LongTensor(query_ids).to(args.device)
    query_ids = query_ids.repeat(len(data['knowledges']), 1)

    knowledge = tokenizer(data['knowledges'], padding=True)
    knowledge_ids = knowledge['input_ids']
    knowledge_ids = torch.LongTensor(knowledge_ids).to(args.device)
    knowledge_ids[knowledge_ids == 0] = args.ignore_index

    return query_ids, knowledge_ids


def train(model, tokenizer, train_data, dev_data, warmup_steps, args):
    accumulate_steps = args.accumulate_steps
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
    for i in tqdm(range(1, epochs + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.train()
        model.zero_grad()
        for index in tqdm(np.random.permutation(len(train_data))):
            query_ids, knowledge_ids = simple_batching_samples(train_data[index], tokenizer, args)
            loss = model(input_ids=query_ids, labels=knowledge_ids).loss
            epoch_loss += loss.item()
            loss /= accumulate_steps
            loss.backward()
            if index % accumulate_steps == 0:
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
            ret_dict = model.generate(
                input_ids=query['input_ids'],
                attention_mask=query['attention_mask'],
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
            scores = sequences_scores / normalize
            max_index = torch.argmax(scores, dim=-1)  # 1
            knowledge_ids = cand_ids[max_index]
            knowledge = tokenizer.decode(knowledge_ids, skip_special_tokens=True)
            d["knowledge"] = knowledge
    result_file = os.path.join(args.result_dir,
                               f"{args.number}_{args.train_data_file.split('/')[-1].split('.')[0]}_{args.LM.split('/')[-1]}_{args.dev_data_file.split('/')[-1]}")
    print(colored('[The result will writing to ' + result_file + '.]', 'blue'))
    with open(result_file, 'w') as f:
        json.dump(data, f, indent=4)


def parse_arguments(parser):
    # Training hyper-parameters
    parser.add_argument('--number', type=str, default='sampled_baseline')
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
    parser.add_argument('--accumulate_steps', type=int, default=10, help="default batch size is 10")
    parser.add_argument('--num_epochs', type=int, default=7, help="Usually we set to 7.")

    # model hyperparameter
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--LM', type=str, default='pretrained_models/t5-large')
    parser.add_argument('--task_prefix', type=str, default="Generating explanations for question with candidates:")
    parser.add_argument('--only_oks', action='store_true')

    # model generation
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--beam_size', type=int, default=6)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--min_length', type=int, default=0)
    parser.add_argument('--length_pen', type=float, default=0.8)
    parser.add_argument('--no_repeat_ngram', type=int, default=3)
    parser.add_argument('--ignore_index', default=-100, type=int)

    args = parser.parse_args()
    if args.only_oks:
        args.number = 'sampled_baseline_only_oks'
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
            new_knowledges = []
            new_oks = []
            for i, ok in enumerate(data['oks']):
                if ok:
                    new_knowledges.append(data['knowledges'][i])
                    new_oks.append(ok)
            data['oks'] = new_oks
            data['knowledges'] = new_knowledges
        if len(data['knowledges']) != 0:
            train_data.append(data)
    print('num of train data:', len(train_data))
    # for data in train_data:
    #     print(data['oks'])
    with open(args.dev_data_file) as f:
        dev_data = json.load(f)
    print('num of dev data:', len(dev_data))
    train(model, tokenizer, train_data, dev_data, 1000, args)


if __name__ == "__main__":
    main()
