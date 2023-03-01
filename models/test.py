
import json

with open(
        'data/csqa/filter_results/unifiedqa-t5-3b_filtered_sampled_knowledge_gpt-j_candidates_answer_explanation.train.csqa.json') as f:
    sampled_data = json.load(f)
with open(
        'data/csqa/filter_results/unifiedqa-t5-3b_filtered_batched_knowledge_gpt-j_candidates_answer_explanation.train.csqa.json') as f:
    batched_data = json.load(f)
with open('data/csqa/knowledge/knowledge_gpt-j_candidates_answer_explanation.train.csqa.json') as f:
    raw_data = json.load(f)
i = 0
for data in sampled_data:
    knowledges_lens = [len(d) for d in data['knowledges']]
    if 0 in knowledges_lens:
        print(data)
    if sum(data['oks']) == 0:
        i += 1
        print(data)
print(i)

for data in batched_data:
    if len(data['knowledge']) == 0:
        print(data)
