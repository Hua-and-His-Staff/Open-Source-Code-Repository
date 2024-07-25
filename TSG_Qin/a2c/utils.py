import glob
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions,is_credit_assignment):
        return super().log_prob(actions).sum(-1, keepdim=True) if is_credit_assignment==False else super().log_prob(actions)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
