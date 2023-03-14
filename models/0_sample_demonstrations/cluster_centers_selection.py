import json
import random
import torch
import numpy as np
import argparse
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


def get_batch_question(args, data, shffule=False):
    if shffule:
        data.sort(key=lambda x: len(x['query']))
    batch_size = args.encoder_batchsize
    train_num = len(data)
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_data = data[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(one_batch_data)
    if shffule:
        random.shuffle(batched_data)
    return batched_data


def get_batch_question_features(args, batch_data):
    global questions_features
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model, cache_dir="pretrained_models/")
    encoder = AutoModel.from_pretrained(args.encoder_model, cache_dir="pretrained_models/").to(args.device)
    for index, data in enumerate(batch_data):
        max_input_len = 0
        input_ids = []
        input_masks = []
        for d in data:
            ids = tokenizer(d['query'])
            input_idx, input_mask = ids['input_ids'], ids['attention_mask']
            max_input_len = max(max_input_len, len(input_idx))
            input_ids.append(input_idx)
            input_masks.append(input_mask)
        for i, input_idx in enumerate(input_ids):
            pad_word_num = max_input_len - len(input_idx)
            input_ids[i].extend([0] * pad_word_num)
            input_masks[i].extend([0] * pad_word_num)
        input_ids = torch.LongTensor(input_ids).to(args.device)
        input_masks = torch.LongTensor(input_masks).to(args.device)
        data_features = encoder(input_ids=input_ids, attention_mask=input_masks, return_dict=True).pooler_output
        if index == 0:
            questions_features = data_features.cpu().detach().numpy()
        else:
            questions_features = np.concatenate((questions_features, data_features.cpu().detach().numpy()))
    return questions_features


def get_questions_features(args, file_name):
    with open(file_name) as f:
        jsdata = json.load(f)
    batched_data = get_batch_question(args, jsdata)
    questions_features = get_batch_question_features(args, batched_data)
    return jsdata, questions_features


def features_normalization(args, questions_features):
    (rows, columns) = questions_features.shape
    if args.normalization == "max_min":
        for i in range(columns):
            questions_features[:, i] = (questions_features[:, i] - np.min(questions_features[:, i])) / (
                    np.max(questions_features[:, i]) - np.min(questions_features[:, i]))
    elif args.normalization == "std":
        for i in range(columns):
            questions_features[:, i] = (questions_features[:, i] - np.mean(questions_features[:, i])) / np.std(
                questions_features[:, i])
    return questions_features


def parse_arguments(parser):
    # 在参数构造器中添加两个命令行参数
    parser.add_argument('--input_file', type=str, default='data/csqa/train.csqa.json')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--encoder_batchsize', type=int, default=48)
    parser.add_argument('--encoder_model', type=str, default="princeton-nlp/sup-simcse-roberta-large")
    parser.add_argument('--normalization', type=str, default='std')
    parser.add_argument('--n_components', type=int, default=100)
    parser.add_argument('--n_clusters', type=int, default=7)
    parser.add_argument('--output_file', type=str, default='knowledge/prompts/csqa_kmeans_candidates_answer_explanation_prompt_test.txt')
    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def main():
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    jsdata, questions_features = get_questions_features(args, args.input_file)
    questions_features = features_normalization(args, questions_features)
    pca = PCA(n_components=args.n_components)
    pca.fit(questions_features)
    print(f"可保留的信息量:{str(round(sum(pca.explained_variance_ratio_), 2) * 100)}%")
    questions_features = pca.fit_transform(questions_features)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42).fit(questions_features)
    print("kmeans收敛轮数：", kmeans.n_iter_)
    print("kmeans簇内差异：", kmeans.inertia_)
    print("kmeans标签：", kmeans.labels_)
    cluster_centers = kmeans.cluster_centers_

    distances = cosine_distances(np.concatenate((questions_features, cluster_centers)))
    for i in range(args.n_clusters):
        distances[i][i] = 1000000
    indices = distances.argmin(axis=1)[:7]
    print(indices)
    kmeans_samples = []
    for i, index in enumerate(indices):
        dis = []
        for j in range(args.n_clusters):
            if distances[j][index] != distances[index][j]:
                raise RuntimeError('eeeeee')
            dis.append(distances[j][index])
        print(f"第{i}个簇中心(index={index})和其他簇中心的差异：{dis}")
        print(jsdata[index])
        kmeans_samples.append(jsdata[index])

    with open(args.output_file, 'w') as f:
        f.write("Generate explanations for the answer to questions. Examples:\n\n")
        for data in kmeans_samples:
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
