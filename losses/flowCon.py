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
  def __init__(self, cfg, device, n_bins = 32):
    self.cfg = cfg
    self.device=device
    self.n_bins = n_bins
    self.device = device
    self.n_pixel = 1280
    self.init_loss = -log(n_bins) * self.n_pixel

  def nllLoss(self, z, logdet, mu, log_sd):
    b_size, _ = z.size()

    # Calculate total log_p
    log_p_total = 0
    log_p_all = torch.zeros((b_size, b_size), dtype=torch.float32, device=self.device)

    # Create mask to select NLL loss elements
    b, d = z.size()
    z = z.view(b, 1, d)
    nll_mask = torch.eye(b, device=self.device).view(b, b, 1)
    nll_mask = nll_mask.repeat(1, 1, d)

    # Square matrix for contrastive loss evaluation      
    log_p_batch = gaussian_log_p(z, mu, log_sd)


    # NLL losses
    log_p_nll = (log_p_batch * nll_mask).sum(dim=(2))
    log_p_nll = log_p_nll.sum(dim=1)

    log_p_all += log_p_batch.sum(dim=(2))

    logdet = logdet.mean()
    loss = self.init_loss + logdet + log_p_nll
    
    return ( 
      (-loss / (log(2) * self.n_pixel)).mean(),
      (log_p_nll / (log(2) * self.n_pixel)).mean(),
      (logdet / (log(2) * self.n_pixel)).mean(), 
      (log_p_all/ (log(2) * self.n_pixel))
      # log_p_nll
  )


  def conLoss(self, log_p_all, labels):
    b, _ = log_p_all.size()
    # tau = torch.index_select(self.temp, 0, labels)
    tau = 0.1
    
    # Create similarity and dissimilarity masks
    off_diagonal = torch.ones((b, b), device=self.device) - torch.eye(b, device=self.device)
    labels = labels.contiguous().view(-1, 1)
    sim_mask = torch.eq(labels, labels.T).float().to(self.device) * off_diagonal
    diff_mask = (1. - sim_mask ) * off_diagonal

    # Get respective log Probablities to compute row-wise pairwise against b*b log_p_all matrix
    diag_logits = (log_p_all * torch.eye(b).to(self.device)).sum(dim=-1)

    # Compute pairwise bhatta coeff. (0.5* (8, 8) + (8, 1))
    # pairwise = (0.5 * (log_p_all.contiguous().view(b, b) + diag_logits.view(b, 1)))
    pairwise = (0.25 * log_p_all.contiguous().view(b, b) + diag_logits.view(b, 1))
    # pairwise = (0.1 * log_p_all.contiguous().view(b, b) + diag_logits.view(b, 1))

    # pairwise = pairwise * off_diagonal
    pairwise_exp = torch.div(torch.exp(
      pairwise - torch.max(pairwise, dim=1, keepdim=True)[0]) + 1e-5, tau)
    # pairwise_exp = torch.div(torch.exp(pairwise), tau)

    pos_count = sim_mask.sum(1)
    pos_count[pos_count == 0] = 1

    log_prob = pairwise_exp - (pairwise_exp.exp() * off_diagonal).sum(-1, keepdim=True).log()

    # compute mean against positive classes
    mean_log_prob_pos = (sim_mask * log_prob).sum(1) / pos_count
    return -mean_log_prob_pos