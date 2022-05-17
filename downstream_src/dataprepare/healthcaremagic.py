import sys
from tqdm import tqdm
import os
import numpy as np
import json
import random
import copy

files = os.listdir(sys.argv[1])
raw_docs = []
for file in files:
    with open(os.path.join(sys.argv[1], file), 'r') as f:
        raw_docs += f.readlines()

raw_samples = ('\n' + ''.join(raw_docs)).split('\nid=')
for idx, item in enumerate(raw_samples):
    if not item:
        continue
    if '\n\nDescription\n' not in item or '\n\nDialogue\n' not in item:
        raise AttributeError(f'Bad Samples: {item}{idx}')
    tmp = item.split('\n\nDescription\n')
    tmp = tmp[:-1] + tmp[-1].split('\n\nDialogue\n')
    raw_samples[idx] = tmp
print(len(raw_samples))
cleaned_samples = []
for idx, item in tqdm(enumerate(raw_samples)):
    if not item:
        continue
    sample = {'id':None, 'tgt':None, 'src':None}
    sample['id'] = idx
    sample['tgt'] = item[1].replace('\n', ' ').strip('Q. ')
    sample['src'] = item[2].replace('\n', ' ').strip(' ')
    cleaned_samples.append(sample)

indices = np.arange(len(cleaned_samples))
random.shuffle(indices)
train_data, dev_data, test_data = [], [], []
for i, idx in enumerate(indices):
    if i < 181122:
        train_data.append(cleaned_samples[idx])
    elif i >= 181122 and i < 181122 + 22641:
        dev_data.append(cleaned_samples[idx])
    else:
        test_data.append(cleaned_samples[idx])

with open('./data/healthcaremagic/train.json', 'w') as f:
    json.dump({'data':train_data}, f, indent=2)

with open('./data/healthcaremagic/dev.json', 'w') as f:
    json.dump({'data':dev_data}, f, indent=2)

with open('./data/healthcaremagic/test.json', 'w') as f:
    json.dump({'data':test_data}, f, indent=2)

print(f'sample numbers, overall {len(indices)}, train {len(train_data)}, dev {len(dev_data)}, test {len(test_data)}.')