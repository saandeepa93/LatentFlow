import torch 
from torch import nn, optim 
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform

import numpy as np
import matplotlib.pyplot as plt
from sys import exit as e
from icecream import ic

# Lightweight datasets
import pickle
from sklearn import datasets

import sys
sys.path.append('.')
from models import RealNVPTabular
from utils import plot_umap, gaussian_log_p, gaussian_sample



class DatasetMoons:
    """ two half-moons """
    def sample(self, n):
        moons = datasets.make_moons(n_samples=n, noise=0.05)[0].astype(np.float32)
        return torch.from_numpy(moons)

mean = torch.zeros((2))
log_sd = torch.zeros((2))
# eps = None
def sample(net, prior, batch_size, eps, cls=None):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    mean = torch.zeros((256, 2))
    log_sd = torch.zeros((256,2))

    with torch.no_grad():
        if cls is not None:
            z = prior.sample((batch_size,), gaussian_id=cls)
        else:
            # z = prior.sample((batch_size,))
            eps = torch.randn((256, 2))
            z = gaussian_sample(eps, mean, log_sd)
        x = net.inverse(z)

        return x


if __name__ == "__main__":
  epochs = 2001
  batch = 256
  feature_dim = 2
  num_mid_channels = 32
  num_blocks = 3
  num_coupling_layers_per_scale = 8
  init_zeros = False

  prior = TransformedDistribution(Uniform(torch.zeros(feature_dim), torch.ones(feature_dim)), SigmoidTransform().inv)

  model = RealNVPTabular(in_dim=feature_dim, hidden_dim=num_mid_channels, num_layers=num_blocks,
                    num_coupling_layers=num_coupling_layers_per_scale, init_zeros=init_zeros)
    # optimizer
  optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) # todo tune WD
  print("number of params: ", sum(p.numel() for p in model.parameters()))

  # DATASET
  d = DatasetMoons()


  model.train()
  for k in range(epochs):
      x = d.sample(batch)

      z, mean, log_sd = model(x)
      sldj = model.logdet()
      # prior_logprob = prior.log_prob(z).view(x.size(0), -1).sum(1)
      prior_logprob = gaussian_log_p(z, mean, log_sd).sum(-1)
      logprob = prior_logprob + sldj
      loss = -torch.sum(logprob) # NLL

      model.zero_grad()
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        if k % 100 ==0:
          ic(mean.mean(), log_sd.mean())
          ic(loss.item())
          x_rec = sample(model, prior, z, batch)
          plt.scatter(x_rec[:,0], x_rec[:,1], c='g', s=5)
          plt.savefig(f"./data/x_{k}.png")
          plt.close()
          plt.scatter(z[:,0], z[:,1], c='r', s=5)
          plt.savefig(f"./data/z_{k}.png")
          plt.close()
      
        if k == epochs - 1:
        # if k == 0:
          ic(mean)
          new_z = torch.cat([z, prior_logprob.view(-1, 1)], dim=-1)
          plot_umap(new_z, dim=3)
