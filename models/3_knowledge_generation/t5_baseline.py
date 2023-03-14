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
    query = f"{args.task_prefix} {data['query']} {','.join(data['cands'])} explanation:"
    query_ids = tokenizer(query, return_tensors='pt').input_ids.to(args.device)
    label_ids = tokenizer(data['answer'], return_tensors='pt').input_ids.to(args.device)
    return query_ids, label_ids


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
    for i in tqdm(range(1, epochs + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.train()
        model.zero_grad()
        for index in tqdm(np.random.permutation(len(train_data))):
            query_ids, label_ids = simple_batching_samples(train_data[index], tokenizer, args)
            loss = model(input_ids=query_ids, labels=label_ids).loss
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
        evaluate_model(model, dev_data, 'dev', tokenizer, args)


def evaluate_model(model, data, name, tokenizer, args):
    print(colored('[Begin evaluating ' + name + '.]', 'green'))
    correct, total = 0, 0
    pbar = tqdm(data)
    with torch.no_grad():
        for d in pbar:
            total += 1
            query = f"{args.task_prefix} {d['query']} {', '.join(d['cands'])} explanation:"
            input_ids = tokenizer(query, return_tensors='pt').input_ids.to(args.device)
            cands = d['cands']
            scores = []
            for i, cand in enumerate(cands):
                labels = tokenizer(cand, return_tensors='pt').input_ids.to(args.device)
                with torch.no_grad():
                    loss = model(input_ids=input_ids, labels=labels).loss.item()
                score = -loss
                scores.append(score)
            scores = torch.Tensor(scores)
            probs = torch.softmax(scores, dim=0)
            p = probs.argmax().item()
            pred = cands[p]
            if d['answer'] == pred:
                correct += 1
            pbar.set_postfix({'acc': correct / total})


def parse_arguments(parser):
    # Training hyper-parameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--train_data_file', type=str,
                        default="data/csqa/train.csqa.json")
    parser.add_argument('--dev_data_file', type=str,
                        default="data/csqa/dev.csqa.json")
    parser.add_argument('--accumulate_steps', type=int, default=10, help="default batch size is 10")
    parser.add_argument('--num_epochs', type=int, default=7, help="Usually we set to 7.")

    # model hyperparameter
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--LM', type=str, default='pretrained_models/t5-large')
    parser.add_argument('--task_prefix', type=str, default="Answer the question with candidates:")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.LM)
    model = T5ForConditionalGeneration.from_pretrained(args.LM)

    with open(args.train_data_file) as f:
        train_data = json.load(f)
    print('num of train data:', len(train_data))
    with open(args.dev_data_file) as f:
        dev_data = json.load(f)
    print('num of dev data:', len(dev_data))
    train(model, tokenizer, train_data, dev_data, 1000, args)


if __name__ == "__main__":
    main()
