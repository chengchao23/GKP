import argparse
import os
import random
import sys
import time
from termcolor import colored
import numpy as np
import torch.cuda
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import json
from torch import optim
from tqdm import tqdm
from KnowledgeGenerator import KnowledgeGenerator


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


def simple_to_ids(data, tokenizer, word_pad_idx, args):
    query = tokenizer(f"{args.task_prefix} {data['query']} {','.join(data['cands'])} explanation:")
    knowledge = tokenizer(data['knowledges'])
    query_ids = query['input_ids']
    knowledge_input_ids = []
    for k in knowledge['input_ids']:
        knowledge_input_ids.append([args.pad_id] + k[:-1])
    knowledge_output_ids = knowledge['input_ids']
    knowledge_lens = [len(k) for k in knowledge['input_ids']]
    max_knowledge_len = max(knowledge_lens)
    for i, knowledge_input_idx in enumerate(knowledge_input_ids):
        pad_word_num = max_knowledge_len - len(knowledge_input_idx)
        knowledge_input_ids[i].extend([word_pad_idx] * pad_word_num)
    for i, knowledge_output_idx in enumerate(knowledge_output_ids):
        pad_word_num = max_knowledge_len - len(knowledge_output_idx)
        knowledge_output_ids[i].extend([word_pad_idx] * pad_word_num)
    query_ids = torch.LongTensor(query_ids).to(args.device)
    query_ids = query_ids.repeat(len(data['knowledges']), 1)
    knowledge_input_ids = torch.LongTensor(knowledge_input_ids).to(args.device)
    knowledge_output_ids = torch.LongTensor(knowledge_output_ids).to(args.device)
    p_n_tag = torch.LongTensor(data['oks']).to(args.device)
    return query_ids, knowledge_input_ids, knowledge_output_ids, p_n_tag


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
        epoch_nll_loss = 0
        epoch_nce_loss = 0
        epoch_pair_loss = 0
        start_time = time.time()
        model.train()
        model.zero_grad()
        for index in tqdm(np.random.permutation(len(train_data))):
            query_ids, knowledge_input_ids, knowledge_output_ids, p_n_tag = \
                simple_to_ids(train_data[index], tokenizer, args.pad_id, args)
            loss = model(query_ids, knowledge_input_ids, knowledge_output_ids, p_n_tag)
            total_loss = loss.get('total_loss')
            epoch_loss += total_loss.item()
            epoch_nll_loss += loss.get('nll_loss').item()
            total_loss /= accumulate_steps
            total_loss.backward()
            if (index+1) % accumulate_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            if len(loss) == 2:
                continue
            if args.loss_func == 'nce_loss' and loss.get('nce_loss') is not None:
                epoch_nce_loss += loss.get('nce_loss').item()
            elif args.loss_func == 'pair_loss' and loss.get('pair_loss') is not None:
                epoch_pair_loss += loss.get('pair_loss').item()
            else:
                raise NotImplementedError(f'{args.loss_func} is not implemented')

        end_time = time.time()
        if args.loss_func == 'nce_loss':
            print("Epoch %d -> Loss: %.5f, Nce_loss: %.5f, Nll_loss: %.5f, Time is %.2fs" % (
                i, epoch_loss, epoch_nce_loss, epoch_nll_loss, end_time - start_time), flush=True)
        elif args.loss_func == 'pair_loss':
            print("Epoch %d -> Loss: %.5f, Pair_loss: %.5f, Nll_loss: %.5f, Time is %.2fs" % (
                i, epoch_loss, epoch_pair_loss, epoch_nll_loss, end_time - start_time), flush=True)
        else:
            raise NotImplementedError(f'{args.loss_func} is not implemented')
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
            query = tokenizer([query], return_tensors='pt').to(args.device)
            result = model.generate(query['input_ids'], query['attention_mask'], args)
            predicted_text = tokenizer.decode(result, skip_special_tokens=True)
            d['knowledge'] = predicted_text
    result_file = os.path.join(args.result_dir,
                               f"{args.number}_{args.train_data_file.split('/')[-1].split('.')[0]}_{args.LM.split('/')[-1]}_{args.dev_data_file.split('/')[-1]}")
    print(colored('[The result will writing to ' + result_file + '.]', 'blue'))
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
    parser.add_argument('--result_dir', type=str, default="data/csqa/generation_results")
    # parser.add_argument('--model_dir', type=str, default="")
    # parser.add_argument('--model_folder', type=str, default="", help="The name to save the model files")
    parser.add_argument('--accumulate_steps', type=int, default=10, help="default batch size is 10")
    parser.add_argument('--num_epochs', type=int, default=7, help="Usually we set to 7.")

    # model hyperparameter
    # parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--loss_func', type=str, default='pair_loss', choices=['nce_loss', 'pair_loss'])
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--LM', type=str, default='pretrained_models/t5-large')
    parser.add_argument('--task_prefix', type=str, default="Generating explanations for question with candidates:")

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
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Prompt knowledge for QA")
    args = parse_arguments(parser)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.LM)
    args.pad_id = tokenizer.pad_token_id
    args.eos_id = tokenizer.eos_token_id
    args.bos_id = tokenizer.bos_token_id

    model = KnowledgeGenerator(args)
    train_data = []
    with open(args.train_data_file) as f:
        jsdata = json.load(f)
    print('num of jsdata:', len(jsdata))
    for data in jsdata:
        if sum(data['oks']) != 0:
            train_data.append(data)
    print('num of train data:', len(train_data))
    with open(args.dev_data_file) as f:
        dev_data = json.load(f)
    print('num of dev data:', len(dev_data))
    train(model, tokenizer, train_data, dev_data, 1000, args)


if __name__ == "__main__":
    main()
