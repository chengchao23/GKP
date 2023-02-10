import json

file_name = 'data/csqa/knowledge/knowledge_gpt-j_candidates_explanation_answer.train.csqa.json'
with open(file_name) as f:
    jsdata = json.load(f)
correct = 0
total = len(jsdata)
total_knowledges = 0
correct_knowledges = 0
new_jsdata = []
for data in jsdata:
    knowledges = data["knowledges"]
    filter_knowledges = []
    answer = []
    for k in knowledges:
        total_knowledges += 1
        truncate_idx = k.find("\nAnswer: ")
        answer.append(k[truncate_idx+len("\nAnswer: "):])
        if k[truncate_idx+len("\nAnswer: "):] == data['answer']:
            correct_knowledges += 1
            filter_knowledges.append(k[:truncate_idx])
    if len(filter_knowledges) > 0:
        data["knowledges"] = filter_knowledges
        new_jsdata.append(data)
    if data['answer'] in answer:
        correct += 1
print(len(new_jsdata))
with open('data/csqa/knowledge/knowledge_gpt-j_candidates_explanation_answer_filter.train.csqa.json', 'w') as f:
    json.dump(new_jsdata, f, indent=4)
print(correct)
print(total)
print(correct_knowledges)
print(total_knowledges)
