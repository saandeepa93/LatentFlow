import torch 
from torch import nn 
from torch.nn import functional as F

from einops import rearrange, reduce, repeat

from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from icecream import ic
from sys import exit as e
import time


def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
  return mean + torch.exp(log_sd) * eps

def bhatta_coeff(z1, z2):
  return 0.5 * (z1 + z2)


class FlowConLoss:
  def __init__(self, cfg, n_bins, device):
    self.cfg = cfg
    self.n_bins = n_bins
    self.device = device
    self.n_pixel = cfg.DATASET.IMG_SIZE * cfg.DATASET.IMG_SIZE * cfg.FLOW.N_CHAN
    self.init_loss = -log(n_bins) * self.n_pixel

    self.temp = torch.tensor(cfg.TRAINING.TEMP).to(device)

    if len(self.temp) != cfg.DATASET.N_CLASS:
      raise ValueError('`TEMP` parameter needs to have the same length as number of class')
    
  def nllLoss(self, out, logdet, means, log_sds):
    b_size, _, _, _ = out[0].size()

    # Calculate total log_p
    log_p_total = 0
    log_p_all = torch.zeros((b_size, b_size), dtype=torch.float32, device=self.device)
    log_p_nll = 0

    for i in range(self.cfg.FLOW.N_BLOCK):
      z = out[i]
      mu = means[i]
      log_sd = log_sds[i]
      
      # Create mask to select NLL loss elements
      b, c, h, w = z.size()
      z = z.view(b, 1, c, h, w)
      nll_mask = torch.eye(b, device=self.device).view(b, b, 1, 1, 1)
      nll_mask = nll_mask.repeat(1, 1, c, h, w)

      # Square matrix for contrastive loss evaluation      
      log_p_batch = gaussian_log_p(z, mu, log_sd)

      # NLL losses
      log_p_nll_block = (log_p_batch * nll_mask).sum(dim=(2, 3, 4))
      log_p_nll_block = log_p_nll_block.sum(dim=1)
      log_p_nll += log_p_nll_block

      log_p_all += log_p_batch.sum(dim=(2, 3, 4))


    logdet = logdet.mean()
    loss = self.init_loss + logdet + log_p_nll
    return ( 
      (-loss / (log(2) * self.n_pixel)).mean(),
      (log_p_nll / (log(2) * self.n_pixel)).mean(),
      (logdet / (log(2) * self.n_pixel)).mean(), 
      (log_p_all/ (log(2) * self.n_pixel))
      # log_p_nll
  )