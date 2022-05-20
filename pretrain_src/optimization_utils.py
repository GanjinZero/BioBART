import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if warmup == 0.0:
        return 1.0
    elif x < warmup:
        return x / warmup
    return 1.0 - x


def warmup_linear_decay_exp(global_step,
                            decay_rate,
                            decay_steps,
                            total_steps,
                            warmup=0.002):
    x = global_step / total_steps
    warmup_end = warmup * total_steps
    if warmup == 0.0:
        return 1.0
    elif x < warmup:
        return x / warmup
    return decay_rate**((global_step - warmup_end) / decay_steps)


def warmup_exp_decay_exp(global_step,
                         decay_rate,
                         decay_steps,
                         total_steps,
                         warmup=0.002,
                         degree=2.0):
    x = global_step / total_steps
    warmup_end = warmup * total_steps
    if warmup == 0.0:
        return 1.0
    elif x < warmup:
        return (x / warmup)**degree
    return decay_rate**((global_step - warmup_end) / decay_steps)


def warmup_exp_decay_poly(global_step,
                          total_steps,
                          warmup=0.002,
                          warm_degree=1.5,
                          degree=2.0):
    x = global_step / total_steps
    if x < warmup:
        return (x / warmup)**warm_degree
    return (1.0 - x)**degree

def warmup_linear_decay_linear(global_step,
                          total_steps,
                          warmup=0.002,
                          ):
    x = global_step / total_steps # x is the propotion of the updated steps wrt total steps
    if x < warmup:
        return x / warmup
    else:
        return 1.0 - (x - warmup)/(1.0 - warmup)