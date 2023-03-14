import json
import random

file_name = 'data/obqa/knowledge/knowledge_gpt-j_candidates_explanation_answer.train.obqa.json'
# file_name = 'data/csqa/knowledge/knowledge_gpt-j_candidates_explanation_answer.train.csqa.json'
random.seed(42)  # Python random module.

with open(file_name) as f:
    jsdata = json.load(f)

# statistics
correct = 0
total_knowledges = 0
correct_knowledges = 0
wrong_knowledges = 0

sampled_data = []
batched_data = []
for data in jsdata:
    knowledges = data["knowledges"]
    oks = []
    preds = []
    new_knowledges = []
    for i, k in enumerate(knowledges):
        total_knowledges += 1
        truncate_idx = k.find("\nAnswer: ")
        if truncate_idx != -1:
            answer = k[truncate_idx + len("\nAnswer: "):]
            knowledge = k[:truncate_idx]
            if len(knowledge) != 0:
                if answer == data['answer']:
                    correct_knowledges += 1
                    oks.append(1)
                    preds.append(answer)
                    new_knowledges.append(knowledge)
                else:
                    wrong_knowledges += 1
                    if random.random() > 0.9:
                        oks.append(0)
                        preds.append(answer)
                        new_knowledges.append(knowledge)
            else:
                print('len(knowledge) == 0')
                print(k)
        else:
            print("truncate_idx == -1")
            print(k)
    if len(new_knowledges) != 0:
        sampled_data.append({"query": data["query"],
                             "cands": data["cands"],
                             "answer": data["answer"],
                             "knowledges": new_knowledges,
                             "oks": oks,
                             "preds": preds})
        for i in range(len(new_knowledges)):
            batched_data.append({"query": data["query"],
                                 "cands": data["cands"],
                                 "answer": data["answer"],
                                 "knowledge": new_knowledges[i],
                                 "ok": oks[i],
                                 "pred": preds[i]})
    else:

        print('len(new_knowledges) == 0')

with open(f'data/obqa/filter_results/filtered_sampled_{file_name.split("/")[-1]}', 'w') as f:
    json.dump(sampled_data, f, indent=4)

with open(f'data/obqa/filter_results/filtered_batched_{file_name.split("/")[-1]}', 'w') as f:
    json.dump(batched_data, f, indent=4)

print('len of jsdata : ', len(jsdata))
print('len of sampled_data : ', len(sampled_data))
print('len of batched_data : ', len(batched_data))
print('num of correct_knowledges : ', correct_knowledges)
print('num of total_knowledges : ', total_knowledges)
print('num of wrong_knowledges : ', wrong_knowledges)
