import os
import json

input_fold = "data/csqa/inference"
output_fold = "data/csqa/step2-data"

input_file = "inference_t5-3b_knowledge_gpt_j.train.csqa.json"
output_file = f"filtered_{input_file}"

with open(os.path.join(input_fold, input_file)) as f:
    jsData = json.load(f)

count = 0
result = []
for data in jsData:
    if len(data['mark_knowledge']) != 0:
        temp = {}
        for knowledge in data['mark_knowledge']:
            temp['query'] = data['query']
            temp['knowledge'] = knowledge
            result.append(temp)
            temp= {}
            count += 1

with open(os.path.join(output_fold, output_file), "w") as f:
    json.dump(result, f, indent=4)
print(count)
