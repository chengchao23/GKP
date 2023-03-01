import argparse
import json

import torch
import transformers
from tqdm import tqdm


def score_for_input(args, tokenizer, model, query, cands, knowledge=None):
    if 'unifiedqa-t5' in args.model_name:  # T5-ft, UnifiedQA, UnifiedQA-ft
        source = f'{query} \\n ' + ','.join(cands)
        if knowledge is not None:
            source = f'{source} \\n {knowledge}'
        targets = cands
    elif 't5' in args.model_name:  # T5
        source = query
        if knowledge is not None:
            source = f"{knowledge} {source}"
        targets = [f'<extra_id_0> {cand} <extra_id_1>' for cand in cands]
    else:
        raise NotImplementedError()
    # if knowledge is not None:
    #     source = f'{knowledge} {query}'
    # f"{query} \\n {', '.join(cands)} \\n {knowledge}"
    # targets = [f'<extra_id_0> {cand} <extra_id_1>' for cand in cands]
    scores = []
    input_ids = tokenizer(source, return_tensors='pt').input_ids.to(args.device)
    for i, cand in enumerate(cands):
        labels = tokenizer(targets[i], return_tensors='pt').input_ids.to(args.device)
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels).loss.item()
        if not args.average_loss:
            loss *= labels.size(1)
        score = -loss
        scores.append(score)
    scores = torch.Tensor(scores)
    probs = torch.softmax(scores, dim=0)
    return scores, probs


def process_item(args, tokenizer, model, item):
    query = item['query'] if 'query' in item else item['question']
    if 'cands' in item:
        cands = item['cands']
    elif args.task == 'csqa2':
        cands = ['yes', 'no']
    else:
        raise Exception('process_item() not implemented for {args.task}!')
    if args.without_knowledge:
        scores, probs = score_for_input(args, tokenizer, model, query, cands)
    else:
        knowledge = item['knowledge']
        scores, probs = score_for_input(args, tokenizer, model, query, cands, knowledge)
    p = probs.argmax().item()
    pred = cands[p]

    item['scores'] = scores.tolist()
    item['probs'] = probs.tolist()
    item['pred'] = pred
    answer = item['answer']
    item['ok'] = 1 if answer == pred else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'])
    parser.add_argument('--task', type=str, default='csqa', choices=['csqa', 'csqa2', 'numersense', 'qasc'])
    parser.add_argument('--model_name', type=str, default='pretrained_models/t5-3b')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--average-loss', action='store_true')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--without_knowledge', action='store_true')
    args = parser.parse_args()
    args.output_path = f'data/{args.task}/inference/{args.model_name.split("/")[-1]}_inference_{args.input_path.split("/")[-1]}'
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir="pretrained_models/")
    if 't5-11b' in args.model_name:
        model.spread_on_devices()
    else:
        model.to(args.device)
    model.eval()
    with open(args.input_path) as f:
        ds = json.load(f)
        if args.n is not None:
            ds = ds[:args.n]

    num, den = 0, 0
    pbar = tqdm(ds)
    for item in pbar:
        process_item(args, tokenizer, model, item)
        if 'ok' in item:
            num += item['ok']
            den += 1
            pbar.set_postfix({'acc': num / den})

    with open(args.output_path, 'w') as f:
        json.dump(ds, f, indent=4)


if __name__ == '__main__':
    '''
        python3 models/inference.py 
        --device cuda:0 
        --task csqa 
        --model_name pretrained_models/t5-3b 
        --input_path data/csqa/step2-result/111003_filtered_inference_t5-3b_knowledge_gpt_j.train.csqa.json  
    '''
    main()
