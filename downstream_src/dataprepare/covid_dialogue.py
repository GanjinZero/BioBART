import json


path_prefix = './COVID-Dialogue/CovidDialog/English-COVID-dialogue-code/data/english'

for split in ['train', 'dev', 'test']:

    srcs = open(f'{path_prefix}/{split}.chat_history', 'r').readlines()
    tgts = open(f'{path_prefix}/{split}.response', 'r').readlines()

    data = [{'src':src.strip('\n'),'tgt':tgt.strip('\n')} for src, tgt in zip(srcs,tgts)]

    with open(f'./data/covid_dialogue/{split}.json', 'w') as f:
        json.dump({'data': data}, f, indent=2)

        