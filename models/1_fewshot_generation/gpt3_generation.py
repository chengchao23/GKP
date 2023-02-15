import argparse
import json
import openai
import pandas as pd
from tqdm import tqdm


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


def request(
        engine,
        prompt,
        temperature,
        top_p,
        n,
        max_tokens,
        stop='\n',
        presence_penalty=0.0,
        frequency_penalty=0.0,
):
    # retry request (handles connection errors, timeouts, and overloaded API)
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            break
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            import time
            time.sleep(60)

    generations = [gen['text'].lstrip() for gen in response['choices']]
    generations = [_ for _ in generations if _ != '']
    return generations


def parse_arguments(parser):
    # 在参数构造器中添加两个命令行参数
    parser.add_argument('--input_path', type=str, default="data/csqa/train.csqa.json")
    parser.add_argument('--prompt_path', type=str, default="knowledge/prompts/csqa_prompt.txt")
    parser.add_argument('--engine', type=str, default="text-davinci-003")
    parser.add_argument('--top_p', default=0.5, type=float)
    parser.add_argument('--temperature', default=0.9, type=float)
    parser.add_argument('--max_tokens', default=50, type=int)
    parser.add_argument('--num_return_sequences', default=10, type=int)
    parser.add_argument('--n', default=None, type=int)
    args = parser.parse_args()
    args.output_path = f"data/csqa/knowledge/knowledge_gpt3_{args.prompt_path.split('/')[-1].split('.')[0]}.{args.input_path.split('/')[-1]}"
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def main():
    openai.api_key = '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'
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    # read examples for inference
    df = pd.read_json(args.input_path)
    df = df[:args.n]
    print("length of data: ", len(df))
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
    for i, row in tqdm(df.iterrows(), total=args.n):
        if 'candidates_answer_explanation_prompt' in args.prompt_path:
            context_string = prompt_format(args.prompt_path, query=row['query'], cands=row['cands'],
                                           answer=row['answer'])
        elif 'candidates_explanation_prompt' in args.prompt_path:
            context_string = prompt_format(args.prompt_path, query=row['query'], cands=row['cands'], answer=None)
        elif 'candidates_explanation_answer_prompt' in args.prompt_path:
            context_string = prompt_format(args.prompt_path, query=row['query'], cands=row['cands'], answer=None)
        else:
            context_string = prompt_format(args.prompt_path, query=row['query'])
        knowledges = request(
            engine=args.engine,
            prompt=context_string,
            temperature=args.temperature,
            top_p=args.top_p,
            n=args.num_return_sequences,
            max_tokens=args.max_tokens)
        row['knowledges'] = list(set(knowledges))
        generated_examples.append(row.to_dict())

    # write results
    with open(args.output_path, 'w') as fo:
        json.dump(generated_examples, fo, indent=4)


if __name__ == '__main__':
    '''
    python models/gpt_generate_knowledge.py \
    --input_path data/csqa/train.csqa.json \
    --prompt_path knowledge/prompts/csqa_prompt.txt 
    '''
    main()
