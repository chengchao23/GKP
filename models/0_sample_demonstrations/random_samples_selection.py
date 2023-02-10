import json
import random
import argparse


def parse_arguments(parser):
    # 在参数构造器中添加两个命令行参数
    parser.add_argument('--input_file', type=str, default='data/csqa/train.csqa.json')
    parser.add_argument('--num_samples', type=int, default=7)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--output_file', type=str, default='knowledge/prompts/csqa_random_candidates_answer_explanation_prompt_{seed}.txt')
    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def main():
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    with open(args.input_file) as f:
        jsdata = json.load(f)
    random.seed(args.random_seed)
    sample_data = random.sample(jsdata, args.num_samples)
    print(sample_data)
    args.output_file = args.output_file.replace('{seed}', str(args.random_seed))
    with open(args.output_file, 'w') as f:
        f.write("Generate explanations for the answer to questions. Examples:\n\n")
        for data in sample_data:
            f.write(f"Input: {data['query']}\n")
            f.write(f"Candidates: {', '.join(data['cands'])}\n")
            f.write(f"Answer: {data['answer']}\n")
            f.write("Explanation: \n\n")
        f.write("Input: {question}\n")
        f.write("Candidates: {cands}\n")
        f.write("Answer: {answer}\n")
        f.write("Explanation: \n")


if __name__ == "__main__":
    main()
