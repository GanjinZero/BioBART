import sys
import json
from tqdm import tqdm
import pandas as pd

with open(sys.argv[1], 'r') as f:
    train_ans = json.load(f)

test_sample_section = []
test_sample_article = []
for idx in tqdm(train_ans):
    question = train_ans[idx]['question']
    for sample in train_ans[idx]['answers']:
        test_sample_section.append({'src':question+' '+train_ans[idx]['answers'][sample]['section'], 'tgt': train_ans[idx]['answers'][sample]["answer_abs_summ"]})
        test_sample_article.append({'src':question+' '+train_ans[idx]['answers'][sample]['article'], 'tgt': train_ans[idx]['answers'][sample]["answer_abs_summ"]})

dev_sample = []
with open(sys.argv[2], 'r') as f:
    raw_sample = json.load(f)
for i in raw_sample:
    dev_sample.append({'src': raw_sample[i]['articles'],'tgt': raw_sample[i]['summary'].strip('<s>').strip('</s>')})

train_sample = []
with open(sys.argv[3], 'r') as f:
    raw_sample = json.load(f)
for i in raw_sample:
    train_sample.append({'src': raw_sample[i]['articles'],'tgt': raw_sample[i]['summary'].strip('<s>').strip('</s>')})

with open('./data/mediqa-ans/test_section.json', 'w') as f:
    json.dump({'data':test_sample_section}, f, indent=2)

with open('./data/mediqa-ans/test_article.json', 'w') as f:
    json.dump({'data':test_sample_article}, f, indent=2)

with open('./data/mediqa-ans/dev.json', 'w') as f:
    json.dump({'data':dev_sample}, f, indent=2)

with open('./data/mediqa-ans/train.json', 'w') as f:
    json.dump({'data':train_sample}, f, indent=2)

print(f'sample numbers, overall {len(train_sample)+len(test_sample_section)+len(dev_sample)}, train {len(train_sample)}, dev {len(dev_sample)}, test {len(test_sample_section)}.')
