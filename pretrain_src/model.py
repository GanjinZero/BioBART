import os
import sys
import numpy as np
import tqdm
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from transformers import BartTokenizer, BartForConditionalGeneration


class BioBARTPTModel:

    def __init__(self, args):
        
        bart_config = AutoConfig.from_pretrained(args.config['bart_model_file'])
        
        self.config = args.config
        self.network = BartForConditionalGeneration.from_pretrained(args.config['bart_model_file'])
        self.tokenizer = BartTokenizer.from_pretrained(args.config['bart_token_file'])
        # self.tokenizer = Tokenizer.from_file(args.config['bart_model_file'] + '/tokenizer.json')

    def set_device(self, device):
        self.device = device

    def save(self, filename: str):
        os.makedirs(filename, exist_ok=True)
        self.network.module.save_pretrained(filename)
        self.tokenizer.save_pretrained(filename)
        return 

    def load(self, model_state_dict: str):
        return self.network.module.load_state_dict(
            torch.load(model_state_dict,
                       map_location=lambda storage, loc: storage))

    def move_batch(self, batch, non_blocking=False):
        return batch.to(self.device, non_blocking)

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def save_bert(self, filename: str):
        return torch.save(self.bert_encoder.state_dict(), filename)

    def to(self, device):
        assert isinstance(device, torch.device)
        self.network.to(device)

    def half(self):
        self.network.half()