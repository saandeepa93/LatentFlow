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


def label_smooth(target, n_classes: int, label_smoothing=0.1):
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target

class FlowConLoss:
  def __init__(self, cfg, device):
    self.cfg = cfg
    self.device=device
    self.n_bins = cfg.FLOW.N_BINS
    self.device = device
    self.n_pixel = cfg.FLOW.IN_FEAT

    self.tau2 = nn.Parameter(torch.tensor(1.5))

    # RAF12
    self.init_loss = -log(self.n_bins) * self.n_pixel

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
      (-loss / (log(2) * self.n_pixel)).mean(), # CONVERTING LOGe to LOG2 |
      (log_p_nll / (log(2) * self.n_pixel)).mean(), #                     v
      (logdet / (log(2) * self.n_pixel)).mean(), 
      (log_p_all/ (log(2) * self.n_pixel))
      # log_p_nll
  )


  def conLoss(self, log_p_all, labels):
    b, _ = log_p_all.size()
    
    tau = 0.1
    
    # Create similarity and dissimilarity masks
    off_diagonal = torch.ones((b, b), device=self.device) - torch.eye(b, device=self.device)
    labels_orig = labels.clone()
    labels = labels.contiguous().view(-1, 1)
    sim_mask = torch.eq(labels, labels.T).float().to(self.device) * off_diagonal
    diff_mask = (1. - sim_mask ) * off_diagonal

    # Get respective log Probablities to compute row-wise pairwise against b*b log_p_all matrix
    diag_logits = (log_p_all * torch.eye(b).to(self.device)).sum(dim=-1)

    # Compute pairwise bhatta coeff. (0.5* (8, 8) + (8, 1))
    pairwise = (self.tau2 * log_p_all.contiguous().view(b, b) + diag_logits.view(b, 1))

    pairwise_exp = torch.div(torch.exp(
      pairwise - torch.max(pairwise, dim=1, keepdim=True)[0]) + 1e-5, tau)
    
    pos_count = sim_mask.sum(1)
    pos_count[pos_count == 0] = 1

    log_prob = pairwise_exp - (pairwise_exp.exp() * off_diagonal).sum(-1, keepdim=True).log()

    # compute mean against positive classes
    mean_log_prob_pos = (sim_mask * log_prob).sum(1) / pos_count
    
    return -mean_log_prob_pos