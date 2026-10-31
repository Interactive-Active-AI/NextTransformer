"""Bounded, identity-centered LoopFormer conditioning, evaluated in FP32."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from nexttransformer.models.backbone import norm


class TimeEmbedding(nn.Module):
    def __init__(self, width, fourier_dim):
        super().__init__()
        self.dim = fourier_dim
        self.mlp = nn.Sequential(nn.Linear(fourier_dim,width),nn.SiLU(),nn.Linear(width,width))

    def forward(self, t):
        half = self.dim//2
        frequencies = torch.exp(-math.log(10000)*torch.arange(half,device=t.device,dtype=torch.float32)/half)
        angles = t.float()[:,None]*frequencies
        features = torch.cat((angles.cos(),angles.sin()),dim=-1)
        if self.dim % 2:features = F.pad(features,(0,1))
        return self.mlp(features)


class LoopConditioner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time = TimeEmbedding(config.width,config.fourier_dim)
        self.delta = TimeEmbedding(config.width,config.fourier_dim)
        self.modulation = nn.ModuleList(nn.Sequential(nn.SiLU(),nn.Linear(config.width,4*config.width,bias=False))
                                       for _ in range(config.core_layers))

    @torch.no_grad()
    def init_weights(self):
        for module in (self.time,self.delta):
            for layer in module.modules():
                if isinstance(layer,nn.Linear):
                    nn.init.normal_(layer.weight,std=.02)
                    nn.init.zeros_(layer.bias)
        for module in self.modulation:nn.init.zeros_(module[-1].weight)

    def forward(self, recursion, batch, device):
        c = self.config
        with torch.autocast(device_type=device.type,enabled=False):
            t = torch.full((batch,),recursion/c.recursions,device=device,dtype=torch.float32)
            delta = torch.full_like(t,1/c.recursions)
            condition = norm(self.time(t)+self.delta(delta))
            # Returned values are offsets: the block uses 1 + offset.
            # Native c_proj starts at zero, so identity gates keep its first
            # update trainable; zero multiplicative gates would kill gradients.
            return [c.loop_modulation_bound*torch.tanh(module(condition)) for module in self.modulation]

    @torch.no_grad()
    def diagnostics(self):
        device = next(self.parameters()).device
        values = torch.cat([x.flatten() for r in range(self.config.recursions)
                            for x in self(r,1,device)])
        return {'loop_modulation_absmax':float(values.abs().max()),
                'loop_modulation_saturated_fraction':float((values.abs()>.95*self.config.loop_modulation_bound).float().mean())}
