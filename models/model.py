import torch 
from torch import nn 

import timm
from sys import exit as e
from icecream import ic

from .realnvp import RealNVPTabular

class LatentModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    # self.effnet = timm.create_model('tf_efficientnet_b0_ns', pretrained=False, num_classes=0, global_pool='')
    self.effnet = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
    self.effnet.classifier = nn.Identity()
    self.effnet.load_state_dict(torch.load('./checkpoints/state_vggface2_enet0_new.pt', map_location=torch.device('cpu')))
    for param in self.effnet.parameters():
      param.requires_grad=False
    self.effnet.eval()

    self.flow = RealNVPTabular(in_dim=cfg.FLOW.IN_FEAT, hidden_dim=cfg.FLOW.MLP_DIM, num_layers=cfg.FLOW.N_FLOW, \
                    num_coupling_layers=cfg.FLOW.N_BLOCK, init_zeros=cfg.FLOW.INIT_ZEROS)
    
  def forward(self, x):
    x = self.effnet(x)
    x, mean, log_sd, logdet = self.flow(x)
    return x, mean, log_sd, logdet



