import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import IntEnum

from icecream import ic 
from sys import exit as e


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AffineConstantFlow(nn.Module):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        
    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def inverse(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
    
    def forward(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)

class Invertible1x1Conv(nn.Module):
    """ 
    As introduced in Glow paper.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim)).to(device)
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim)).to(self.L.device)
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def inverse(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x

class iSequential(torch.nn.Sequential):

    def inverse(self, y):
        for module in reversed(self._modules.values()):
            assert hasattr(module,'inverse'), '{} has no inverse defined'.format(module)
            y = module.inverse(y)
        return y

    def logdet(self):
        log_det = 0
        for module in self._modules.values():
            assert hasattr(module,'logdet'), '{} has no logdet defined'.format(module)
            log_det += module.logdet()
        return log_det

    def reduce_func_singular_values(self,func):
        val = 0
        for module in self._modules.values():
            if hasattr(module,'reduce_func_singular_values'):
                val += module.reduce_func_singular_values(func)
        return val


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1
    TABULAR = 2
    HORIZONTAL = 3
    VERTICAL = 4
    Quadrant = 5
    SubQuadrant = 6
    Center = 7


def checkerboard_mask(dim, reverse, device):
    mask = torch.zeros((dim), device=device, requires_grad=False)
    if reverse:
        mask[::2] = 1
    else:
        mask[1::2] = 1
    return mask.view(1, -1)

class MaskCheckerboard:
    def __init__(self, reverse_mask):
        self.type = MaskType.CHECKERBOARD
        self.reverse_mask = reverse_mask

    def mask(self, x):
        # self.b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
        self.b = checkerboard_mask(x.size(1), self.reverse_mask, device=x.device)
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)
    
    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)

    def get_valid_half(self, x):
        if not self.reverse_mask:
            return x[:, 1::2]
        else:
            return x[:, ::2]


class MaskTabular:
    def __init__(self, reverse_mask):
        self.type = MaskType.TABULAR
        self.reverse_mask = reverse_mask

    def mask(self, x):
        dim = x.size(1)
        split = dim // 2
        self.b = torch.zeros((1, dim), dtype=torch.float).to(x.device)
        
        if self.reverse_mask:
            self.b[:, split:] = 1.
            # x_id = x[:, split:]
            # x_change = x[:, :split]
        else:
            self.b[:, :split] = 1.
            # x_id = x[:, :split]
            # x_change = x[:, split: ]
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)
    
    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)
      
    def get_valid_half(self, x):
      dim = x.size(1)
      split = dim // 2
      if not self.reverse_mask:
          return x[:, :split]
      else:
        return x[:, split:]

class RescaleTabular(nn.Module):
    def __init__(self, D):
        super(RescaleTabular, self).__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        x = self.weight * x
        return x

class CouplingLayerBase(nn.Module):
    """Coupling layer base class in RealNVP.
    
    must define self.mask, self.st_net, self.rescale
    """

    def _get_st(self, x):
        x_id, x_change = self.mask.mask(x)
        st = self.st_net(x_id)
        s, t = st.chunk(2, dim=1)
        s = self.rescale(torch.tanh(s))

        return s, t, x_id, x_change

    def forward(self, x, sldj=None, reverse=True):
        s, t, x_id, x_change = self._get_st(x)
        s, t = self.mask.mask_st_output(s, t)

        exp_s = s.exp()
        if torch.isnan(exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = (x_change + t) * exp_s
        self._logdet = s.view(s.size(0), -1).sum(-1)
        if self.mask.type == MaskType.SubQuadrant:
            # DEBUG!!!!!!!
           self._logdet = self.mask.reshape_logdet(self._logdet) 
        x = self.mask.unmask(x_id, x_change)

        # LEARNED PRIOR
        x_change_valid = self.mask.get_valid_half(x_id)
        mean, log_sd = self.prior(x_change_valid).chunk(2, 1)
        
        return x, mean, log_sd, self._logdet

    def inverse(self, y):
        s, t, x_id, x_change = self._get_st(y)
        s, t = self.mask.mask_st_output(s, t)
        exp_s = s.exp()
        inv_exp_s = s.mul(-1).exp()
        if torch.isnan(inv_exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = x_change * inv_exp_s - t
        self._logdet = -s.view(s.size(0), -1).sum(-1)
        x = self.mask.unmask(x_id, x_change)

        return x

    def logdet(self):
        return self._logdet


class ZeroLinear(nn.Module):
  def __init__(self, in_dim, out_dim, mid_dim=256):
    super().__init__()
    self.linear = nn.Linear(in_dim, out_dim)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    # self.linear2 = nn.Linear(out_dim, out_dim)
    # self.linear2.weight.data.zero_()
    # self.linear2.bias.data.zero_()
    # self.linear2 = nn.Linear(mid_dim, out_dim)
    # self.swish = nn.ReLU()
  
  def forward(self, input):
    out = self.linear(input)
    
    return out

# class ScaleNet(nn.Module):
#     def __init__(self, in_dim, mid_dim, dropout, num_layers, init_zeros):
#         super().__init__()
        
#         self.dropout_flg = dropout
#         self.dropout = nn.Dropout(0.5)
#         self.l1 = nn.Linear(in_dim, mid_dim)
#         self.silu = nn.SiLU()
#         self.bn1 = nn.BatchNorm1d(mid_dim)
#         self.bn2 = nn.BatchNorm1d(in_dim * 2)
#         self.l2_list = nn.ModuleList()
#         for _ in range(num_layers):
#             self.l2_list.append(nn.Linear(mid_dim, mid_dim))
#         self.last = ZeroLinear(mid_dim, in_dim*2)
    
#     def forward(self, x):
#         x = self.l1(x)
#         x = self.silu(x)
#         if self.dropout:
#             x = self.dropout(x)
#         for l2 in self.l2_list:
#             x = l2(x)
#             x = self.bn1(x)
#             x = self.silu(x)
#         x = self.last(x)
#         x = self.bn2(x)
#         x = self.silu(x)
#         return x

class CouplingLayerTabular(CouplingLayerBase):

    def __init__(self, in_dim, mid_dim, num_layers, mask, init_zeros=False, dropout=False):
        
        super(CouplingLayerTabular, self).__init__()
        self.mask = mask
        self.st_net = nn.Sequential(nn.Linear(in_dim, mid_dim),
                                    nn.SiLU(),
                                    nn.Dropout(.5) if dropout else nn.Sequential(),
                                    *self._inner_seq(num_layers, mid_dim),
                                    ZeroLinear(mid_dim, in_dim*2)
                                    )
        # self.st_net = ScaleNet(in_dim, mid_dim, dropout, num_layers, init_zeros)
        self.prior = ZeroLinear(in_dim//2, in_dim*2)
        
        if init_zeros:
                # init w zeros to init a flow w identity mapping
                torch.nn.init.zeros_(self.st_net[-1].weight)
                torch.nn.init.zeros_(self.st_net[-1].bias)

        self.rescale = nn.utils.weight_norm(RescaleTabular(in_dim))

    @staticmethod
    def _inner_seq(num_layers, mid_dim):
        res = []
        for _ in range(num_layers):
            res.append(nn.Linear(mid_dim, mid_dim))
            # UPDATE
            # res.append(nn.Dropout(.5))
            # UPDATE
            res.append(nn.SiLU())
        return res

  
class Flow(nn.Module):
    def __init__(self, in_dim, i, hidden_dim, num_layers, init_zeros, dropout):
        super().__init__()

        # self.actnorm = ActNorm(in_dim)
        # self.invconv = Invertible1x1Conv(in_dim)
        self.coupling = CouplingLayerTabular(
                    in_dim, hidden_dim, num_layers, \
                    MaskTabular(reverse_mask=bool(i%2)),  \
                    # MaskCheckerboard(reverse_mask=bool(i%2)),  \
                    init_zeros=init_zeros, dropout=dropout
                )

    def forward(self, x):
        # x, det1 = self.actnorm(x)
        # x, det2 = self.invconv(x)
        x, mean, log_sd, det3  = self.coupling(x)
        logdet = det3
        return x, mean, log_sd, logdet
    
    def inverse(self, input):
        input = self.coupling.inverse(input)
        # input = self.invconv.inverse(input)
        # input = self.actnorm.inverse(input)
        return input

class RealNVPBase(nn.Module):

    def forward(self,x):
      # return self.body(x)
      logdet_all = 0
      for body in self.body:
        x, mean, log_sd, log_det = body(x)
        logdet_all += log_det
      return x, mean, log_sd, logdet_all

    def logdet(self):
        return self.body.logdet()

    def inverse(self, z):
        # return self.body.inverse(z)
        for flow in self.body[::-1]:
            z = flow.inverse(z)
        return z

    def nll(self,x,y=None,label_weight=1.):
        z = self(x)
        logdet = self.logdet()
        z = z.reshape((z.shape[0], -1))
        prior_ll = self.prior.log_prob(z, y,label_weight=label_weight)
        nll = -(prior_ll + logdet)
        return nll



class RealNVPTabular(RealNVPBase):

    def __init__(self, cfg, in_dim=2, num_coupling_layers=6, hidden_dim=256, 
                 num_layers=2, init_zeros=False, dropout=False):

        super(RealNVPTabular, self).__init__()
        
        self.body = nn.ModuleList()
        for i in range(num_coupling_layers):
            # in_dimf = in_dim if i == 0 else in_dim * 2
            self.body.append(Flow(in_dim, i, hidden_dim, num_layers, init_zeros, dropout))
        
        # self.body = iSequential(*[
        #                 CouplingLayerTabular(
        #                     in_dim, hidden_dim, num_layers, MaskTabular(reverse_mask=bool(i%2)), init_zeros=init_zeros, dropout=dropout)
        #                     # in_dim, hidden_dim, num_layers, MaskCheckerboard(reverse_mask=bool(i%2)), init_zeros=init_zeros, dropout=dropout)
        #                 for i in range(num_coupling_layers)
        #             ])
        