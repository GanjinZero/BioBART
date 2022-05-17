import sys
from tqdm import tqdm
import json
import numpy as np
import random

with open(sys.argv[1], 'r') as f:
    raw_docs = f.readlines()

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
    if i < 24851:
        train_data.append(cleaned_samples[idx])
    elif i >= 24851 and i < 24851 + 3105:
        dev_data.append(cleaned_samples[idx])
    else:
        test_data.append(cleaned_samples[idx])

with open('./data/icliniq/train.json', 'w') as f:
    json.dump({'data':train_data}, f, indent=2)

with open('./data/icliniq/dev.json', 'w') as f:
    json.dump({'data':dev_data}, f, indent=2)

with open('./data/icliniq/test.json', 'w') as f:
    json.dump({'data':test_data}, f, indent=2)

print(f'sample numbers, overall {len(indices)}, train {len(train_data)}, dev {len(dev_data)}, test {len(test_data)}.')