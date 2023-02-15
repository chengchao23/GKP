import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import json
import torch
from pathlib import Path
from typing import List, Optional
import argparse
from transformers import GPT2LMHeadModel, AutoTokenizer, GPTJForCausalLM


def prompt_format(prompt_path: str, query: str, cands: list = None, answer: str = None):
    with open(prompt_path) as f:
        context_string = f.read().strip('\n')
    if query is not None:
        context_string = context_string.replace('{question}', query)
    if cands is not None:
        candidates = ", ".join(cands)
        context_string = context_string.replace('{cands}', candidates)
    if answer is not None:
        context_string = context_string.replace('{answer}', answer)
    return context_string


def parse_arguments(parser):
    # 在参数构造器中添加两个命令行参数
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--input_path', type=str, default="data/csqa/train.csqa.json")
    parser.add_argument('--output_path', type=str, default="data/csqa/knowledge/knowledge_gpt-j.train.csqa.json")
    parser.add_argument('--prompt_path', type=str, default="knowledge/prompts/csqa_prompt.txt")
    parser.add_argument('--pretrained_models', type=str, default="pretrained_models/gpt-j-6B-fp16")
    parser.add_argument('--top_p', default=0.5, type=float)
    parser.add_argument('--top_k', default=50, type=int)
    parser.add_argument('--temperature', default=0.9, type=float)
    parser.add_argument('--max_new_tokens', default=50, type=int)
    parser.add_argument('--num_return_sequences', default=10, type=int)
    parser.add_argument('--n', default=None, type=int)
    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def main():
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    # read examples for inference
    eval_df = pd.read_json(args.input_path)
    eval_df = eval_df[:args.n]
    print("length of data: ", len(eval_df))
    # model initialization
    print("use ", args.device)
    print("loading model....")
    if 'gpt-j' in args.pretrained_models:
        if 'fp16' in args.pretrained_models:
            model = GPTJForCausalLM.from_pretrained(args.pretrained_models, torch_dtype=torch.float16).to(args.device)
        else:
            model = GPTJForCausalLM.from_pretrained(args.pretrained_models).to(args.device)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_models).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_models)
    print("loading model completed!")
    # generate knowledge!
    if 'candidates_answer_explanation_prompt' in args.prompt_path:
        print('use candidates and answer to generate explanation to help question')
    elif 'candidates_explanation_prompt' in args.prompt_path:
        print('use candidates to generate explanation to help question')
    elif 'candidates_explanation_answer_prompt' in args.prompt_path:
        print('use candidates to generate explanation and answer to help question')
    else:
        print('generate explanation with only question')
    generated_examples = []
    for i, row in tqdm(eval_df.iterrows(), total=args.n):
        if 'candidates_answer_explanation_prompt' in args.prompt_path:
            context_string = prompt_format(args.prompt_path, query=row['query'], cands=row['cands'], answer=row['answer'])
        elif 'candidates_explanation_prompt' in args.prompt_path:
            context_string = prompt_format(args.prompt_path, query=row['query'], cands=row['cands'], answer=None)
        elif 'candidates_explanation_answer_prompt' in args.prompt_path:
            context_string = prompt_format(args.prompt_path, query=row['query'], cands=row['cands'], answer=None)
        else:
            context_string = prompt_format(args.prompt_path, query=row['query'])
        context_string_ids = tokenizer.encode(context_string, return_tensors="pt").to(args.device)
        sample_outputs = model.generate(
            context_string_ids,
            do_sample=True,
            temperature=args.temperature,
            max_length=context_string_ids.shape[1] + args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=50256)
        generated_outputs = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
        knowledges = []
        for line in generated_outputs:
            truncate_idx = line.find("\n\n", len(context_string))
            if truncate_idx != -1 and len(line[len(context_string) + 1: truncate_idx]) != 0:
                knowledges.append(line[len(context_string) + 1: truncate_idx])
        row['knowledges'] = list(set(knowledges))
        generated_examples.append(row.to_dict())

    # write results
    with open(args.output_path, 'w') as fo:
        json.dump(generated_examples, fo, indent=4)


if __name__ == '__main__':
    '''
    python models/gpt_generate_knowledge.py \
    --device cuda:0 \
    --input_path data/csqa/train.csqa.json \
    --output_path data/csqa/knowledge/knowledge_gpt-j.train.csqa.json \
    --prompt_path knowledge/prompts/csqa_prompt.txt \
    --pretrained_models pretrained_models/gpt-j-6B-fp16
    '''
    main()
