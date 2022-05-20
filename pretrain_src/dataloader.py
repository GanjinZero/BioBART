import os
import random
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from datagen import BioBARTPretrainDataCreator, TokenInstance
from itertools import groupby

import threading
import queue
import time


class AsyncWorker(threading.Thread):
    def __init__(self, dataloaders, dataset_picker):

        threading.Thread.__init__(self)
        self.req_queue = queue.Queue()
        self.ret_queue = queue.Queue()
        self.dataloaders = dataloaders
        self.dataset_picker = dataset_picker
        self.prefetch_idx = 3
        for i in range(self.prefetch_idx):
            self.req_queue.put(dataset_picker[i])

    def run(self):
        while True:
            dataset_type = self.req_queue.get(block=True)
            if dataset_type is None:
                break
            batch = next(self.dataloaders[dataset_type])
            self.req_queue.task_done()
            self.ret_queue.put(batch)

    def get(self):
        batch = self.ret_queue.get()
        self.ret_queue.task_done()
        return batch

    def prefetch(self):
        if self.prefetch_idx < len(self.dataset_picker):
            self.req_queue.put(self.dataset_picker[self.prefetch_idx])
            self.prefetch_idx += 1

    def stop(self):
        self.req_queue.put(None)


def map_to_torch(encoding):
    encoding = torch.LongTensor(encoding)
    encoding.requires_grad_(False)
    return encoding

def get_random_partition(data_directory, index):
    partitions = [
        os.path.join(data_directory, x) for x in os.listdir(data_directory)
    ]
    partitions = sorted(partitions)
    i = index % len(partitions)
    return partitions[i]

def padding_to_maxlength(ids, max_length):
    cur_len = len(ids)
    len_diff = max_length-len(ids)
    return ids + [1] * len_diff, [1] * cur_len + [0] * len_diff

class PreTrainingDataset(Dataset):
    def __init__(self,
                 tokenizer ,
                 folder: str,
                 logger,
                 max_seq_length,
                 index,
                 token_nosing_prob,
                 max_predictions_per_seq: int = 300):
        self.tokenizer = tokenizer
        self.dir_path = folder
        self.max_seq_length = max_seq_length
        self.len = 0
        self.max_predictions_per_seq = max_predictions_per_seq
        self.masked_lm_prob = token_nosing_prob
        self.vocab_words = list(tokenizer.get_vocab().keys())

        path = get_random_partition(self.dir_path, index)

        logger.info(f"Loading Pretraining Data from {path}")
        start = time.time()
        self.data = BioBARTPretrainDataCreator.load(path)
        self.len = len(self.data)

        logger.info(
            f"Data Loading Completed for Pretraining Data from {path} with {self.len} samples took {time.time()-start:.2f}s."
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        i = index % self.len

        instance = self.data.instances[i]
        return self.create_training_instance(instance)

    def create_training_instance(self, instance: TokenInstance):
        
        token_x, token_y, is_rotate = instance.get_values()

        x = []
        y = []

        y = token_y + ['</s>']

        # Get Masked LM predictions
        noised_tokens, number_masked_tokens = self.create_noised_input(token_x)
        # noised_tokens = token_x

        x.append('<s>')
        x = x + noised_tokens
        x.append('</s>')
    
        input_ids, attn_mask = padding_to_maxlength(self.tokenizer.convert_tokens_to_ids(x), self.max_seq_length)
        labels, decoder_attn_mask = padding_to_maxlength(self.tokenizer.convert_tokens_to_ids(y), self.max_seq_length)
        # input_ids, attn_mask = padding_to_maxlength(self.tokenizer.encode(x, is_pretokenized=True, add_special_tokens = False).ids[:self.max_seq_length], self.max_seq_length)
        # labels, decoder_attn_mask = padding_to_maxlength(self.tokenizer.encode(y, is_pretokenized=True, add_special_tokens = False).ids[:self.max_seq_length], self.max_seq_length)

        return [map_to_torch(input_ids), map_to_torch(labels), map_to_torch(attn_mask), map_to_torch(decoder_attn_mask)]

    def create_noised_input(self, tokens_x):
        masked_number = 0
        noised_x = []
        noised_sent = []
        for i, sent in enumerate(tokens_x):
            j = 0
            noised_sent = []
            while j < len(sent):
                if random.random() < self.masked_lm_prob:
                    num_tokens_to_mask = np.random.poisson(lam = 3)
                    masked_number += num_tokens_to_mask
                    if num_tokens_to_mask > 0:
                        noised_sent.append('<mask>')
                        j += num_tokens_to_mask
                    else:
                        noised_sent.append(sent[j])
                        noised_sent.append('<mask>')
                        j += 1
                else:
                    noised_sent.append(sent[j])
                    j += 1
            noised_x.append(noised_sent)
        # random.shuffle(noised_x)
        noised_x = sum(noised_x, [])

        return noised_x, masked_number

class BARTDatasetProviderInterface:
    def get_shard(self, index, shuffle=True):
        raise NotImplementedError

    def release_shard(self, index):
        raise NotImplementedError

    def prefetch_shard(self, index):
        raise NotImplementedError

    def get_batch(self, batch_iter):
        raise NotImplementedError

    def prefetch_batch(self):
        raise NotImplementedError

class BioBARTDatasetProvider(BARTDatasetProviderInterface):
    def __init__(self, args):
        self.tokenizer = args.tokenizer
        self.refresh_bucket_size = args.refresh_bucket_size
        self.datasampler = RandomSampler if args.local_rank == -1 else DistributedSampler
        self.num_workers = args.config['training']['num_workers']
        self.token_nosing_prob = args.token_nosing_prob

        # Initialize dataset paths
        self.dataset_paths = []
        self.dataset_paths.append(
            os.path.join(args.data_path_prefix,
                        args.config["data"]["datasets"]["pubmed_pretrain_dataset"]
                        )
                                 )

        self.max_seq_length = args.max_seq_length
        self.max_predictions_per_seq = args.max_predictions_per_seq

        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
        self.local_rank = args.local_rank
        self.global_rank = dist.get_rank()
        self.world_size = 1 if self.local_rank == -1 else dist.get_world_size()
        self.logger = args.logger

        self.dataloaders = {}
        self.dataset_iterator = []

        # Configure asynchronous data loading
        self.async_dataloading = 'async_worker' in args.config['training']
        self.async_worker = None

        if self.global_rank == 0:
            self.logger.info(
                f"BioBARTDatasetProvider - Initialization:  async data loading {self.async_dataloading}"
            )

    def get_shard(self, index, shuffle=True):
        datalengths = []
        batches_per_dataset = []
        for i, dataset_path in enumerate(self.dataset_paths):
            pretrain_dataset = PreTrainingDataset(
                tokenizer=self.tokenizer,
                folder=dataset_path,
                logger=self.logger,
                max_seq_length=self.max_seq_length,
                index=index,
                token_nosing_prob=self.token_nosing_prob,
                max_predictions_per_seq=self.max_predictions_per_seq)

            datalengths.append(len(pretrain_dataset))
            batches_per_dataset.append(
                self._get_effective_batch(len(pretrain_dataset)))
            self.dataloaders[i] = self._get_dataloader(pretrain_dataset)

        dataset_batches = []
        for i, batch_count in enumerate(batches_per_dataset):
            dataset_batches.extend([i] * batch_count)

        # shuffle
        if shuffle:
            random.shuffle(dataset_batches)

        self.dataset_iterator = []
        for dataset_batch_type in dataset_batches:
            self.dataset_iterator.extend([dataset_batch_type] *
                                         self.gradient_accumulation_steps *
                                         self.refresh_bucket_size)

        if self.async_dataloading:
            self.async_worker = AsyncWorker(self.dataloaders,
                                            self.dataset_iterator)
            self.async_worker.start()

        return self.dataset_iterator, sum(datalengths)

    def release_shard(self, index):
        if self.async_dataloading:
            self.async_worker.stop()

    def prefetch_shard(self, index):
        pass

    def get_batch(self, batch_iter):
        if self.async_dataloading:
            return self.async_worker.get()
        return next(self.dataloaders[batch_iter])

    def prefetch_batch(self):
        if self.async_dataloading:
            self.async_worker.prefetch()

    def _get_dataloader(self, dataset: Dataset):
        return (
            x
            for x in DataLoader(dataset,
                                batch_size=self.train_micro_batch_size_per_gpu,
                                sampler=self.datasampler(dataset),
                                num_workers=self.num_workers))

    def _get_effective_batch(self, total):
        return total // self.world_size // self.train_micro_batch_size_per_gpu // self.gradient_accumulation_steps // self.refresh_bucket_size