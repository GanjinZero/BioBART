import sys
import json
from tqdm import tqdm
import pandas as pd

with open(sys.argv[1], 'r') as f:
    train_ans = json.load(f)

train_sample = []
for idx in tqdm(train_ans):
    question = train_ans[idx]['question']
    for sample in train_ans[idx]['answers']:
        train_sample.append({'src':question+' '+train_ans[idx]['answers'][sample]['section'], 'tgt': train_ans[idx]['answers'][sample]["answer_abs_summ"]})
        train_sample.append({'src':question+' '+train_ans[idx]['answers'][sample]['article'], 'tgt': train_ans[idx]['answers'][sample]["answer_abs_summ"]})


with open(sys.argv[2]+'_question.txt', 'r') as f:
    questions = {int(line.strip().split('||', maxsplit=1)[0]):line.strip().split('||', maxsplit=1)[1]  for line in f.readlines()}
with open(sys.argv[2]+'_summary.txt', 'r') as f:
    summaries = {int(line.strip().split('||', maxsplit=1)[0]):line.strip().split('||', maxsplit=1)[1]  for line in f.readlines()}
answers = pd.read_excel(sys.argv[2]+'_answer.xlsx', header=0)

test_sample = []
for id in questions:
    inp = questions[id] + ' ' + ' '.join(list(answers[answers['question_id']==id]['Answer']))
    out = summaries[id]
    test_sample.append({'src':inp, 'tgt':out})


with open(sys.argv[3]+'_question.txt', 'r') as f:
    questions = {int(line.strip().split('||', maxsplit=1)[0]):line.strip().split('||', maxsplit=1)[1]  for line in f.readlines()}
with open(sys.argv[3]+'_summary.txt', 'r') as f:
    summaries = {int(line.strip().split('||', maxsplit=1)[0]):line.strip().split('||', maxsplit=1)[1]  for line in f.readlines()}
answers = pd.read_excel(sys.argv[3]+'_answer.xlsx', header=0)

dev_sample = []
for id in questions:
    inp = questions[id] + ' ' + ' '.join(list(answers[answers['QuestionID']==id]['Answer']))
    out = summaries[id]
    dev_sample.append({'src':inp, 'tgt':out})


with open('./data/mediqa-mas/train.json', 'w') as f:
    json.dump({'data':train_sample}, f, indent=2)

with open('./data/mediqa-mas/dev.json', 'w') as f:
    json.dump({'data':dev_sample}, f, indent=2)

with open('./data/mediqa-mas/test.json', 'w') as f:
    json.dump({'data':test_sample}, f, indent=2)

print(f'sample numbers, overall {len(train_sample)+len(test_sample)+len(dev_sample)}, train {len(train_sample)}, dev {len(dev_sample)}, test {len(test_sample)}.')
