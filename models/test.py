import json
import jsonlines

with jsonlines.open(
        'data/openbookqa/train.jsonl') as data:
    train_data = []
    for d in data:
        query = d['question']['stem']
        cands = [cand['text'] for cand in d['question']['choices']]
        answer = cands[ord(d['answerKey']) - 65]
        train_data.append({'query': query,
                           'cands': cands,
                           'answer': answer})
with open('data/openbookqa/train.obqa.json', 'w') as f:
    json.dump(train_data, f, indent=4)

with jsonlines.open(
        'data/openbookqa/dev.jsonl') as data:
    dev_data = []
    for d in data:
        query = d['question']['stem']
        cands = [cand['text'] for cand in d['question']['choices']]
        answer = cands[ord(d['answerKey']) - 65]
        dev_data.append({'query': query,
                         'cands': cands,
                         'answer': answer})
with open('data/openbookqa/dev.obqa.json', 'w') as f:
    json.dump(dev_data, f, indent=4)

with jsonlines.open(
        'data/openbookqa/test.jsonl') as data:
    test_data = []
    for d in data:
        query = d['question']['stem']
        cands = [cand['text'] for cand in d['question']['choices']]
        answer = cands[ord(d['answerKey']) - 65]
        test_data.append({'query': query,
                          'cands': cands,
                          'answer': answer})
with open('data/openbookqa/test.obqa.json', 'w') as f:
    json.dump(test_data, f, indent=4)
