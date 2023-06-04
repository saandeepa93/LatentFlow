import torch 
from torch import nn 
from torch.nn import functional as F

from einops import rearrange, reduce, repeat

from math import log, pi, exp
import os
import numpy as np
from scipy import linalg as la
from icecream import ic
from sys import exit as e
import time
from utils import gaussian_log_p


class FocalLoss(nn.Module):
  def __init__(self, cfg, args):
    super().__init__()
    self.cfg = cfg
    self.gamma = 0.5
    self.n_pixel = cfg.FLOW.IN_FEAT

    dist_path = f"./data/distributions/{args.config}"
    mu = torch.load(os.path.join(dist_path, "mu.pt"))
    log_sd = torch.load(os.path.join(dist_path, "log_sd.pt"))
    self.mu =  torch.stack(mu)
    self.log_sd =  torch.stack(log_sd)

  def calc_likelihood(self, z):
    z = rearrange(z, 'b d -> b 1 d')
    log_p_batch = gaussian_log_p(z, self.mu, self.log_sd)
    log_p_all = log_p_batch.sum(dim=(2))
    return log_p_all/ (log(2) * self.n_pixel)

  def forward(self, input, target, z):
    # log_pt = input.log_softmax(1)
    ce = F.cross_entropy(input, target, reduction='none')
    weights_vec = F.softmax(self.calc_likelihood(z), dim=-1)
    target_vec = F.one_hot(target, num_classes = self.cfg.DATASET.N_CLASS)
    target_vec = target_vec.type(input.dtype)
    weights = (target_vec * weights_vec).sum(-1)

    f1 = torch.pow(weights, self.gamma) 
    focal = f1 * ce
    # loss_tmp = torch.einsum('bc...,bc...->b...', (target_vec, focal))
    return focal.mean()
    




