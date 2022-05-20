import sys
import argparse
import logging
import torch.distributed as dist
from tensorboardX import SummaryWriter
import os
import torch

SUMMARY_WRITER_DIR_NAME = 'runs'


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

def report_step_metrics(args, lr, loss, step, data_sample_count):
    ##### Record the LR against global_step on tensorboard #####
    if (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1):
        args.summary_writer.add_scalar(f'Train/lr', lr, step)

        args.summary_writer.add_scalar(f'Train/Samples/train_loss', loss,
                                       data_sample_count)

        args.summary_writer.add_scalar(f'Train/Samples/lr', lr,
                                       data_sample_count)
    ##### Recording  done. #####

    if (step + 1) % args.print_steps == 0 and master_process(args):
        print('bing_bert_progress: step={}, loss={}, lr={}, sample_count={}'.
              format(step + 1, loss, lr, data_sample_count))

def master_process(args):
    return (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1)

def report_lamb_coefficients(args, optimizer):
    if master_process(args):
        if (args.fp16 and args.use_lamb):
            #print("Lamb Coeffs", optimizer.optimizer.get_lamb_coeffs())
            lamb_coeffs = optimizer.optimizer.get_lamb_coeffs()
            lamb_coeffs = np.array(lamb_coeffs)
            if lamb_coeffs.size > 0:
                args.summary_writer.add_histogram(f'Train/lamb_coeffs',
                                                  lamb_coeffs, global_step)

def get_sample_writer(name, base=".."):
    """Returns a tensorboard summary writer
    """
    return SummaryWriter(
        log_dir=os.path.join(base, SUMMARY_WRITER_DIR_NAME, name))

def save_checkpoint_model(PATH, ckpt_id, model, last_global_step,
                     last_global_data_samples, args, **kwargs):

    checkpoint_state_dict = {
        'last_global_step': last_global_step,
        'last_global_data_samples': last_global_data_samples
    }
    checkpoint_state_dict.update(kwargs)
    logger.info(f"Saving model checkpoint to {PATH}")

    model.save(os.path.join(PATH, ckpt_id))

    status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(PATH, ckpt_id)

    logging.info(f"Success {status_msg}")

    return

class Logger():
    def __init__(self, cuda=False):
        self.logger = logging.getLogger(__name__)
        self.cuda = cuda

    def info(self, message, *args, **kwargs):
        if (self.cuda and dist.get_rank() == 0) or not self.cuda:
            self.logger.info(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Required_parameter
    parser.add_argument(
        "--config-file",
        "--cf",
        help="pointer to the configuration file of the experiment",
        type=str,
        required=True)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written."
    )

    parser.add_argument(
        "--token_nosing_prob",
        default=0.15,
        type=float,
        help="The probability that a token is masked."
    )

    # Optional Params
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        "--max_pred",
        default=80,
        type=int,
        help=
        "The maximum number of masked tokens in a sequence to be predicted.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument(
        "--do_lower_case",
        default=True,
        action='store_true',
        help=
        "Whether to lower case the input text. True for uncased models, False for cased models."
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--use_pretrain',
                        default=False,
                        action='store_true',
                        help="Whether to use Bert Pretrain Weights or not")

    parser.add_argument(
        '--refresh_bucket_size',
        type=int,
        default=1,
        help=
        "This param makes sure that a certain task is repeated for this time steps to \
                            optimise on the back propogation speed with APEX's DistributedDataParallel"
    )

    parser.add_argument('--finetune',
                        default=False,
                        action='store_true',
                        help="Whether to finetune only")

    parser.add_argument(
        '--lr_schedule',
        type=str,
        default='LE',
        help=
        'Choices LE, EE, EP (L: Linear, E: Exponetial, P: Polynomial warmup and decay)'
    )

    parser.add_argument('--lr_offset',
                        type=float,
                        default=0.0,
                        help='Offset added to lr.')

    parser.add_argument(
        '--load_training_checkpoint',
        '--load_cp',
        type=str,
        default=None,
        help=
        "This is the path to the TAR file which contains model+opt state_dict() checkpointed."
    )

    parser.add_argument(
        '--load_checkpoint_id',
        '--load_cp_id',
        type=str,
        default=None,
        help='Checkpoint identifier to load from checkpoint path')

    parser.add_argument(
        '--job_name',
        type=str,
        default=None,
        help="This is the path to store the output and TensorBoard results.")

    parser.add_argument(
        '--rewarmup',
        default=False,
        action='store_true',
        help='Rewarmup learning rate after resuming from a checkpoint')

    parser.add_argument(
        '--max_steps',
        type=int,
        default=sys.maxsize,
        help=
        'Maximum number of training steps of effective batch size to complete. Which is override by total_update_step in model config'
    )

    parser.add_argument(
        '--max_steps_per_epoch',
        type=int,
        default=sys.maxsize,
        help=
        'Maximum number of training steps of effective batch size within an epoch to complete.'
    )

    parser.add_argument('--print_steps',
                        type=int,
                        default=100,
                        help='Interval to print training details.')
    
    parser.add_argument('--save_steps',
                        type=int,
                        default=1000,
                        help='Interval to save the checkpoints.')

    parser.add_argument(
        '--data_path_prefix',
        type=str,
        default="./",
        help=
        "Path to prefix data loading, helpful for AML and other environments")

    parser.add_argument(
        '--validation_data_path_prefix',
        type=str,
        default=None,
        help=
        "Path to prefix validation data loading, helpful if pretraining dataset path is different"
    )

    parser.add_argument('--deepspeed_transformer_kernel',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed transformer kernel to accelerate.')

    parser.add_argument(
        '--stochastic_mode',
        default=False,
        action='store_true',
        help='Use stochastic mode for high-performance transformer kernel.')

    parser.add_argument(
        '--ckpt_to_save',
        nargs='+',
        type=int,
        help=
        'Indicates which checkpoints to save, e.g. --ckpt_to_save 160 161, by default all checkpoints are saved.'
    )

    parser.add_argument(
        '--attention_dropout_checkpoint',
        default=False,
        action='store_true',
        help=
        'Use DeepSpeed transformer kernel memory optimization to checkpoint dropout output.'
    )
    parser.add_argument(
        '--normalize_invertible',
        default=False,
        action='store_true',
        help=
        'Use DeepSpeed transformer kernel memory optimization to perform invertible normalize backpropagation.'
    )
    parser.add_argument(
        '--gelu_checkpoint',
        default=False,
        action='store_true',
        help=
        'Use DeepSpeed transformer kernel memory optimization to checkpoint GELU activation.'
    )
    parser.add_argument('--deepspeed_sparse_attention',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed sparse self attention.')

    parser.add_argument('--progressive_layer_drop',
                        default=False,
                        action='store_true',
                        help="Whether to enable progressive layer dropping or not")

    return parser


def is_time_to_exit(args, epoch_steps=0, global_steps=0):
    return (epoch_steps >= args.max_steps_per_epoch) or \
            (global_steps >= args.max_steps)