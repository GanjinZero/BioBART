import sys
import pandas as pd
from tqdm import tqdm
import json

meqsum = pd.read_excel(sys.argv[1], header=0)
train_set = []
for idx in tqdm(range(len(meqsum))):
    sample = {'src':None, 'tgt':None}
    sample['src'] = meqsum.iloc[idx]['CHQ'].strip('\n').replace('\n', ' ')
    sample['tgt'] = meqsum.iloc[idx]['Summary'].strip('\n').replace('\n', ' ')
    train_set.append(sample)

task1 = pd.read_excel(sys.argv[2], header=0)
test_set = []
for idx in tqdm(range(len(task1))):
    sample = {'src':None, 'tgt':None}
    sample['src'] = task1.iloc[idx]['NLM Question'].strip('\n').replace('\n', ' ')
    sample['tgt'] = task1.iloc[idx]['Summary'].strip('\n').replace('\n', ' ')
    test_set.append(sample)

task1 = pd.read_excel(sys.argv[3], header=0)
dev_set = []
for idx in tqdm(range(len(task1))):
    sample = {'src':None, 'tgt':None}
    sample['src'] = task1.iloc[idx]['NLM Question'].strip('\n').replace('\n', ' ')
    sample['tgt'] = task1.iloc[idx]['Summary'].strip('\n').replace('\n', ' ')
    dev_set.append(sample)


with open('./data/mediqa-qs/train.json', 'w') as f:
    json.dump({'data':train_set}, f, indent=2)

with open('./data/mediqa-qs/dev.json', 'w') as f:
    json.dump({'data':dev_set}, f, indent=2)

with open('./data/mediqa-qs/test.json', 'w') as f:
    json.dump({'data':test_set}, f, indent=2)

print(f'sample numbers, overall {len(train_set)+len(test_set)+len(dev_set)}, train {len(train_set)}, dev {len(dev_set)}, test {len(test_set)}.')
