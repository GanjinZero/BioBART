import os
import sys
import time
import logging
import numpy as np
import random
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.sampler import RandomSampler, SequentialSampler
# from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from optimization_utils import warmup_linear, warmup_linear_decay_exp, warmup_exp_decay_exp, warmup_exp_decay_poly, warmup_linear_decay_linear
from utils import get_argument_parser, is_time_to_exit, Logger, get_sample_writer, report_step_metrics, report_lamb_coefficients, save_checkpoint_model

# from bing_bert_dataset_provider import BingBertDatasetProvider
# from nvidia_bert_dataset_provider import NvidiaBertDatasetProvider

from transformers import BartTokenizer
# from tokenizers import Tokenizer
from datagen import BioBARTPretrainDataCreator, TokenInstance
from dataloader import BioBARTDatasetProvider
from model import BioBARTPTModel
import deepspeed

global_step = 0
global_data_samples = 0
last_global_step_from_restore = 0
all_step_time = 0.0

def checkpoint_model(PATH, ckpt_id, model, epoch, last_global_step,
                     last_global_data_samples, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        'epoch': epoch,
        'last_global_step': last_global_step,
        'last_global_data_samples': last_global_data_samples
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.network.save_checkpoint(PATH, ckpt_id,
                                            checkpoint_state_dict)
    status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(PATH, ckpt_id)
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return

def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # no cuda mode is not supported
    args.no_cuda = False

    return args


def construct_arguments():

    args = get_arguments()

    # Prepare Logger
    logger = Logger(cuda=torch.cuda.is_available() and not args.no_cuda)
    args.logger = logger
    config = json.load(open(args.config_file, 'r', encoding='utf-8'))

    # choose dataset and training config based on the given sequence length
    seq_len = str(args.max_seq_length)

    datasets = config["data"]["mixed_seq_datasets"][seq_len]
    del config["data"]["mixed_seq_datasets"]
    training = config["mixed_seq_training"][seq_len]
    del config["mixed_seq_training"]
    config["data"]["datasets"] = datasets
    config["training"] = training
    args.config = config

    args.max_steps = args.config["training"]["total_training_steps"]

    args.job_name = config['name'] if args.job_name is None else args.job_name
    print("Running Config File: ", args.job_name)
    # Setting the distributed variables
    print("Args = {}".format(args))

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    args.saved_model_path = os.path.join(args.output_dir, "saved_models/",
                                         args.job_name)

    # args.n_gpu = 1

    tokenizer = BartTokenizer.from_pretrained(config["bart_token_file"])
    args.tokenizer = tokenizer

    # Set validation dataset path
    if args.validation_data_path_prefix is None:
        logging.warning(
            'Skipping validation because validation_data_path_prefix is unspecified'
        )

    # Issue warning if early exit from epoch is configured
    if args.max_steps < sys.maxsize:
        logging.warning(
            'Early training exit is set after {} global steps'.format(
                args.max_steps))

    if args.max_steps_per_epoch < sys.maxsize:
        logging.warning('Early epoch exit is set after {} global steps'.format(
            args.max_steps_per_epoch))

    return args

def prepare_model_optimizer(args):
    # Initialize torch distributed
    deepspeed.init_distributed(dist_backend='nccl')
    args.local_rank = int(os.environ['LOCAL_RANK'])

    # Loading Model
    model = BioBARTPTModel(args)

    # Optimizer parameters
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
    model.network, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model.network,
        model_parameters=optimizer_grouped_parameters)

    # Overwrite application configs with DeepSpeed config
    args.train_micro_batch_size_per_gpu = model.network.train_micro_batch_size_per_gpu()
    args.gradient_accumulation_steps = model.network.gradient_accumulation_steps()

    # Set DeepSpeed info
    args.local_rank = model.network.local_rank
    args.device = model.network.device
    model.set_device(args.device)
    args.fp16 = model.network.fp16_enabled()
    args.use_lamb = (model.network.optimizer_name() ==
                     deepspeed.runtime.config.LAMB_OPTIMIZER
                     or model.network.optimizer_name() ==
                     deepspeed.runtime.config.ONEBIT_LAMB_OPTIMIZER)

    # Prepare Summary Writer and saved_models path
    if dist.get_rank() == 0:
        summary_writer = get_sample_writer(name=args.job_name,
                                           base=args.output_dir)
        args.summary_writer = summary_writer
        os.makedirs(args.saved_model_path, exist_ok=True)

    return model, optimizer


def prepare_optimizer_parameters(args, model):

    config = args.config
    deepspeed_config = json.load(open(args.deepspeed_config, 'r', encoding='utf-8'))

    param_optimizer = list(model.network.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.deepspeed_transformer_kernel:
        no_decay = no_decay + [
            'attn_nw', 'attn_nb', 'norm_w', 'norm_b', 'attn_qkvb', 'attn_ob',
            'inter_b', 'output_b'
        ]
    if "weight_decay" in config["training"].keys():
        weight_decay = config["training"]["weight_decay"]
    else:
        weight_decay = 0.01

    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in param_optimizer
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        weight_decay
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    return optimizer_grouped_parameters

def update_learning_rate(args, config, current_global_step, optimizer):

    global last_global_step_from_restore

    global_step_for_lr = current_global_step - last_global_step_from_restore

    if args.lr_schedule == "EE":
        # print(f'LR Schedule is {args.lr_schedule} EE')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_exp_decay_exp(
                global_step_for_lr, config["training"]["decay_rate"],
                config["training"]["decay_step"],
                config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    elif args.lr_schedule == "EP":
        # print(f'LR Schedule is {args.lr_schedule} EP')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_exp_decay_poly(
                global_step_for_lr, config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    elif args.lr_schedule == "LE":
        # print(f'LR Schedule is {args.lr_schedule} LE')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_linear_decay_exp(
                global_step_for_lr, config["training"]["decay_rate"],
                config["training"]["decay_step"],
                config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    elif args.lr_schedule == "LL":
        # print(f'LR Schedule is {args.lr_schedule} LL')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_linear_decay_linear(
                global_step_for_lr,
                config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    lr_this_step += args.lr_offset

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step

    return lr_this_step


def train(args,
          index,
          model,
          optimizer,
          pretrain_dataset_provider):
          
    global global_step
    global global_data_samples
    global last_global_step_from_restore
    global all_step_time

    dataset_iterator, total_length = pretrain_dataset_provider.get_shard(index)
    current_data_sample_count = global_data_samples

    config = args.config
    logger = args.logger
    logger.info(
        f'worker-{dist.get_rank()}: begin epoch {index+1} current_sample_count {current_data_sample_count} shard_length {total_length} global_data_samples {global_data_samples}'
    )

    pretrain_dataset_provider.prefetch_shard(index + 1)

    model.train()

    epoch_step = 0
    rounds = 20
    step_counts = 0

    for _, batch_index in enumerate(tqdm(dataset_iterator, smoothing=1)):
        # try:
        step_start = time.time()
        batch = pretrain_dataset_provider.get_batch(batch_index)
        batch = tuple(t.to(args.device) for t in batch)  # Move to GPU

        # Calculate forward pass
        input_ids = batch[0]
        labels = batch[1]
        attention_mask = batch[2]
        decoder_attention_mask = batch[3]
        # print(input_ids.shape)
        # print(labels.shape)
        # print(attention_mask.shape)
        # print(decoder_attention_mask.shape)
        # print(input_ids[0])
        # print(labels[0])
        # input()
        output = model.network(input_ids = input_ids,
                                attention_mask = attention_mask,
                                labels = labels,
                                decoder_attention_mask = decoder_attention_mask,
                                return_dict = True
                                )
        loss = output.loss
        unscaled_loss = loss.item()
        current_data_sample_count += (args.train_micro_batch_size_per_gpu *
                                        dist.get_world_size())

        # Prefetch training data
        pretrain_dataset_provider.prefetch_batch()

        model.network.backward(loss)

        loss = None

        if model.network.is_gradient_accumulation_boundary():
            if args.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = update_learning_rate(
                    args, config, global_step, optimizer)

            report_step_metrics(args, lr_this_step, unscaled_loss,
                                global_step, current_data_sample_count)

            model.network.step()

            report_lamb_coefficients(args, optimizer)
            global_step += 1
            epoch_step += 1
        else:
            # Call DeepSpeed engine step on micro steps
            model.network.step()
            ## in step() method when is_gradient_accumulation_boundary() returns false
            ## the weights are not updated and the other state tracking params are updated


        current_global_step = global_step - last_global_step_from_restore

        # if current_global_step % args.save_steps == 0:
        #     save_checkpoint_model(PATH=args.saved_model_path,
        #                           ckpt_id='checkpoint_global_step_{}'.format(global_step),
        #                           model=model,
        #                           last_global_step=global_step,
        #                           last_global_data_samples=global_data_samples,
        #                           args = args,
        #                           )

        if is_time_to_exit(args=args,
                           epoch_steps=epoch_step,
                           global_steps=current_global_step):
            print(
                f'Warning: Early epoch termination due to max steps limit, epoch step ={epoch_step}, global step = {current_global_step}, epoch = {index+1}'
            )
            break
        step_time = time.time() - step_start
        all_step_time += step_time
        if global_step % rounds == 0 and global_step != 0 and model.network.is_gradient_accumulation_boundary(
        ) and dist.get_rank() == 0:
            one_step_bs = args.train_micro_batch_size_per_gpu * args.gradient_accumulation_steps * dist.get_world_size(
            ) * rounds
            print(' At step {}, the throughput is {:2f} Samples/s'.format(
                global_step * args.gradient_accumulation_steps,
                one_step_bs / all_step_time),
                  flush=True)
            all_step_time = 0.0

    pretrain_dataset_provider.release_shard(index)
    global_data_samples = current_data_sample_count


def run(args, model, optimizer, start_epoch):

    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger

    pretrain_dataset_provider = BioBARTDatasetProvider(args)
    for index in range(start_epoch, config["training"]["num_epochs"]):
        logger.info(f"Training Epoch: {index + 1}")
        pre = time.time()

        train(args, index, model, optimizer, pretrain_dataset_provider)


        if index % 100 == 0:
            save_checkpoint_model(PATH=args.saved_model_path,
                                    ckpt_id='checkpoint_global_step_{},epoch_index_{}'.format(global_step, index+1),
                                    model=model,
                                    last_global_step=global_step,
                                    last_global_data_samples=global_data_samples,
                                    args = args,
                                    )

        post = time.time()
        logger.info(f"Time for shard {index + 1}: {post-pre} seconds")

        current_global_step = global_step - last_global_step_from_restore
        if is_time_to_exit(args=args, global_steps=current_global_step):
            print(
                f'Warning: Early training termination due to max steps limit, epoch={index+1}, global_step={current_global_step}'
            )
            save_checkpoint_model(PATH=args.saved_model_path,
                                    ckpt_id='checkpoint_global_step_{},epoch_index_{}'.format(global_step, index+1),
                                    model=model,
                                    last_global_step=global_step,
                                    last_global_data_samples=global_data_samples,
                                    args = args,
                                    )
            break

def main():
    start = time.time()
    args = construct_arguments()
    model, optimizer = prepare_model_optimizer(args)
    start_epoch = 0
    # if not None in [args.load_training_checkpoint, args.load_checkpoint_id]:
    #     start_epoch = load_checkpoint(args, model)
    run(args, model, optimizer, start_epoch)
    elapsed = time.time() - start
    logger = args.logger
    logger.info(f"Elapsed time: {elapsed} seconds")


if __name__ == "__main__":
    main()