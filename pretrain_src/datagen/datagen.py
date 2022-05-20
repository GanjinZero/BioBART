from tqdm import tqdm
from typing import Tuple
import pickle
import random
import numpy as np
import os
import copy
from multiprocessing import Process
from transformers import BartTokenizer

def truncate_input_sequence(document, max_num_tokens):

    total_length = len(sum(document, []))
    if total_length <= max_num_tokens:
        return document
    else:
        tokens_to_trunc = total_length - max_num_tokens
        while tokens_to_trunc > 0:
            if len(document[-1]) >= tokens_to_trunc:
                document[-1] = document[-1][:len(document[-1])-tokens_to_trunc]
                tokens_to_trunc = 0
            else:
                tokens_to_trunc -= len(document[-1])
                document = document[:-1]
        return document

class TokenInstance:
    """ This TokenInstance is a obect to have the basic units of data that should be
        extracted from the raw text file and can be consumed by any BERT like model.
    """
    def __init__(self, tokens_x, tokens_y, is_rotate, lang="en"):
        self.tokens_x = tokens_x
        self.tokens_y = tokens_y
        self.is_rotate = is_rotate 
        self.lang = lang

    def get_values(self):
        return (self.tokens_x, self.tokens_y, self.is_rotate)

    def get_lang(self):
        return self.lang

class PretrainingDataCreator:
    def __init__(self,
                 path,
                 tokenizer,
                 max_seq_length,
                 readin = 2000000,
                 dupe_factor = 5,
                 small_seq_prob = 0.1,
                 ):
        
        raise NotImplementedError('not implemented!!!')

    def __len__(self):
        return self.len

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def create_training_instance(self, index):
        
        raise NotImplementedError('not implemented')


class BioBARTPretrainDataCreator(PretrainingDataCreator):
    def __init__(self,
                 path,
                 tokenizer,
                 max_seq_length: int = 512,
                 readin: int = 2000000,
                 dupe_factor: int = 5,
                 sentence_permute_prob: float = 0.8,
                 small_seq_prob: float = 0.1):

        self.dupe_factor = dupe_factor
        self.max_seq_length = max_seq_length
        self.small_seq_prob = small_seq_prob
        self.sentence_permute_prob = sentence_permute_prob

        documents = []
        instances = []
        new_doc = False
        with open(path, 'r', encoding='utf-8') as fd:
            for i, line in enumerate(tqdm(fd)):
                line = line.strip('\n')
                contents = line.split('\t')

                if contents[0] == 'title':
                    new_doc = True
                    abstract_part = ''

                if new_doc:
                    if 'Abstract' in contents[0] and len(contents)>1:
                        abstract_part = abstract_part + ' ' + contents[1]
                    elif contents[0] == '':
                        document = []
                        if len(abstract_part) > 10:
                            abstract_part = abstract_part.strip().split('. ')
                            for sent in abstract_part:
                                document.append(tokenizer.tokenize(sent.strip(' ').strip('.')+'.')) ## donot need bos and eos tokens
                            documents.append(document)
                        new_doc = False

        documents = [x for x in documents if x]

        self.documents = documents

        for _ in range(self.dupe_factor):
            for index in range(len(self.documents)):
                instances.extend(self.create_training_instance(index))

        random.shuffle(instances)
        self.instances = instances
        self.len = len(self.instances)

        # self.documents = None
        documents = None
    
    def create_training_instance(self, index):

        instances = []
        # Need to add [bos] + [eos] tokens
        max_num_tokens = self.max_seq_length - 2

        document = truncate_input_sequence(copy.deepcopy(self.documents[index]), max_num_tokens)

        y = sum(document, [])
        is_rotate = False

        instances.append(TokenInstance(document, y, int(is_rotate)))

        return instances
    
    def merge_documents(self, instance_lists):

        self.instances = sum(instance_lists, [])
        self.len = len(self.instances)

def process_data(begin, end):
    print('Start Processs:',begin,'->',end)
    path = './pubmed_txt_abstract'
    input_files = os.listdir(path)
    tokenizer = BartTokenizer.from_pretrained('./bart-base')
    instance_lists = []
    document_lists = []
    for i in range(begin, min(end,len(input_files))):
        d_c = BioBARTPretrainDataCreator(
            path = os.path.join(path, input_files[i]),
            tokenizer = tokenizer,
            max_seq_length = 512,
            readin = 2000000,
            dupe_factor = 1,
            sentence_permute_prob = 1,
        )
        instance_lists.append(d_c.instances)
        # document_lists.append(d_c.documents)
    d_c.merge_documents(instance_lists)
    d_c.save('./pretrain_data/bart_base_512_cased/pubmed_'+str(begin//10).rjust(3,'0')+'.pkl')

if __name__ == '__main__':
    # pocess_data(0,10)
    threads = []
    path = './'
    input_files = os.listdir(path)
    for idx in range(0, len(input_files), 10):
        if len(threads) == 32:
            for t in threads:
                t.join()
            threads = []
        else:
            thread = Process(target=process_data, args=(idx, idx+10,))
            thread.start()
            threads.append(thread)
    
    for t in threads:
        t.join()

            


