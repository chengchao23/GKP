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


def simple_batching_samples(data, tokenizer, word_pad_idx, args):
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
    knowledge_input_ids = []
    max_knowledge_input_len = 0
    knowledge_output_ids = []
    max_knowledge_output_len = 0

    input_ids = []
    input_attention_mask = []
    max_input_len = 0

    p_n_tag = []

    for idx, data in enumerate(batch_data):
        p_n_tag.append(data['ok'])
        if 't5' in args.LM:
            query = tokenizer(f"{args.task_prefix} {data['query']} {', '.join(data['cands'])} explanation:")
            knowledge = tokenizer(data['knowledge'])
            query_idx = query['input_ids']
            knowledge_input_idx = [args.pad_id] + knowledge['input_ids'][:-1]
            knowledge_output_idx = knowledge['input_ids']
            max_query_len = max(max_query_len, len(query_idx))
            max_knowledge_input_len = max(max_knowledge_input_len, len(knowledge_input_idx))
            max_knowledge_output_len = max(max_knowledge_output_len, len(knowledge_output_idx))
            query_ids.append(query_idx)
            knowledge_input_ids.append(knowledge_input_idx)
            knowledge_output_ids.append(knowledge_output_idx)
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
        for i, knowledge_input_idx in enumerate(knowledge_input_ids):
            pad_word_num = max_knowledge_input_len - len(knowledge_input_idx)
            knowledge_input_ids[i].extend([word_pad_idx] * pad_word_num)
        for i, knowledge_output_idx in enumerate(knowledge_output_ids):
            pad_word_num = max_knowledge_output_len - len(knowledge_output_idx)
            knowledge_output_ids[i].extend([word_pad_idx] * pad_word_num)

        query_ids = torch.LongTensor(query_ids).to(args.device)
        knowledge_input_ids = torch.LongTensor(knowledge_input_ids).to(args.device)
        knowledge_output_ids = torch.LongTensor(knowledge_output_ids).to(args.device)
        p_n_tag = torch.LongTensor(p_n_tag).to(args.device)
        return query_ids, knowledge_input_ids, knowledge_output_ids, p_n_tag
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
    batch_size = args.batch_size
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
    if 'batched' in args.train_data_file:
        batched_train_data = batching_list_instances(batch_size, train_data)
    elif 'sampled' in args.train_data_file:
        batched_train_data = train_data
    else:
        raise NotImplementedError()
    # evaluate_model(model, dev_data, 'dev', tokenizer, args)
    lowest_loss = 0
    for i in tqdm(range(1, epochs + 1), desc="Epoch"):
        epoch_loss = 0
        epoch_nll_loss = 0
        epoch_nce_loss = 0
        epoch_pair_loss = 0
        start_time = time.time()
        model.train()
        model.zero_grad()
        for index in tqdm(np.random.permutation(len(batched_train_data))):
            if 'batched' in args.train_data_file:
                query_ids, knowledge_input_ids, knowledge_output_ids, p_n_tag = simple_batching(
                    batched_train_data[index], tokenizer, args.pad_id, args)
                if sum(p_n_tag) == 0:
                    continue
            elif 'sampled' in args.train_data_file:
                query_ids, knowledge_input_ids, knowledge_output_ids, p_n_tag = simple_batching_samples(
                    batched_train_data[index], tokenizer, args.pad_id, args)
            else:
                raise NotImplementedError()
            loss = model(query_ids, knowledge_input_ids, knowledge_output_ids, p_n_tag)
            total_loss = loss.get('total_loss')
            epoch_loss += total_loss.item()
            epoch_nll_loss += loss.get('nll_loss').item()
            if args.loss_func == 'nce_loss' and loss.get('nce_loss') is not None:
                epoch_nce_loss += loss.get('nce_loss').item()
            elif args.loss_func == 'pair_loss' and loss.get('pair_loss') is not None:
                epoch_pair_loss += loss.get('pair_loss').item()

            if 'batched' in args.train_data_file:
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            elif 'sampled' in args.train_data_file:
                total_loss /= accumulate_steps
                total_loss.backward()
                if index % accumulate_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
            else:
                raise NotImplementedError()
        end_time = time.time()
        if args.without_contrastive:
            print("Epoch %d -> Loss: %.5f, Nll_loss: %.5f, Time is %.2fs" % (
                i, epoch_loss, epoch_nll_loss, end_time - start_time), flush=True)
        else:
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
    batch_idx = 0
    # generate by sample.
    predicted_result = []
    with torch.no_grad():
        for d in tqdm(data):
            tmp = {}
            query = f"{args.task_prefix} {d['query']} {', '.join(d['cands'])} explanation:"
            query = tokenizer([query], return_tensors='pt').to(args.device)
            result = model.generate(query['input_ids'], query['attention_mask'], args)
            predicted_text = tokenizer.decode(result, skip_special_tokens=True)
            tmp['query'] = d['query']
            tmp['knowledge'] = predicted_text
            tmp['cands'] = d['cands']
            tmp['answer'] = d['answer']
            predicted_result.append(tmp)
            batch_idx += 1
    result_file = os.path.join(args.result_dir,
                               f"{args.number}_{args.train_data_file.split('/')[-1].split('.')[0]}_{args.LM.split('/')[-1]}_{args.dev_data_file.split('/')[-1]}")
    print(colored('[The result will writing to' + result_file + '.]', 'blue'))
    with open(result_file, 'w') as f:
        json.dump(predicted_result, f, indent=4)


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
    parser.add_argument('--batch_size', type=int, default=12, help="default batch size is 12")
    parser.add_argument('--accumulate_steps', type=int, default=10, help="default batch size is 8")
    parser.add_argument('--num_epochs', type=int, default=5, help="Usually we set to 5.")

    # model hyperparameter
    # parser.add_argument('--alpha', type=float, default=0.5)
    # parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--loss_func', type=str, default='pair_loss', choices=['nce_loss', 'pair_loss'])
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--LM', type=str, default='pretrained_models/t5-large')
    parser.add_argument('--task_prefix', type=str, default="Generating explanations for question with candidates:")
    parser.add_argument('--pad_id', type=int, default=0)
    parser.add_argument('--without_contrastive', action='store_true')
    parser.add_argument('--early_stop', type=bool, default=True)

    # model generation
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--beam_size', type=int, default=10)
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
        if 'batched' in args.train_data_file:
            for data in jsdata:
                # 删除知识长度为0的知识
                if len(data['knowledge']) == 0:
                    continue
                # 去除样本中的错误知识
                if args.without_contrastive and data['ok'] == 0:
                    continue
                train_data.append(data)
            print('num of train data:', len(train_data))
        elif 'sampled' in args.train_data_file:
            for data in jsdata:
                # if len(data['knowledges']) != 0:
                if sum(data['oks']) != 0 and len(data['knowledges']) != 0:
                    # 删除知识长度为0的知识
                    for i, k in enumerate(data['knowledges']):
                        if len(k) == 0:
                            del data['knowledges'][i]
                            del data['oks'][i]
                    # 去除样本中的错误知识
                    if args.without_contrastive:
                        new_knowledge = []
                        new_oks = []
                        for i, ok in enumerate(data['oks']):
                            if ok != 0:
                                new_knowledge.append(data['knowledges'][i])
                                new_oks.append(data['oks'][i])
                        data['knowledges'] = new_knowledge
                        data['oks'] = new_oks
                    # 与在forward中使用positive_mask进行mask之后计算出loss无明显区别：
                    # 在样本中剔除: 0.49  vs 在上述办法中mask: 0.489
                    train_data.append(data)
            print('num of train data:', len(train_data))
        else:
            raise NotImplementedError()
    with open(args.dev_data_file) as f:
        dev_data = json.load(f)
    print('num of dev data:', len(dev_data))
    train(model, tokenizer, train_data, dev_data, 1000, args)


if __name__ == "__main__":
    '''
    python3 models/main.py \
    --number 0 \
    --device cuda:1 \
    --train_data_dir \
    data/csqa/filter_results/t5-3b_filtered_knowledge_gpt-j_candidates_answer_explanation.train.csqa.json \
    --batch_size 12
    '''
    main()
