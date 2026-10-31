"""Halting distributions and depth sampling, independent of the LM wrapper."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.models.backbone import norm


def sinusoidal_coordinates(length, width, step, device, dtype):
    inv = 10000.**(-torch.arange(0,width,2,device=device,dtype=torch.float32)/width)
    pos = torch.arange(length,device=device,dtype=torch.float32)[:,None]*inv
    depth = torch.tensor(float(step),device=device)*inv
    a = torch.stack((pos.sin(),pos.cos()),-1).flatten(-2)
    b = torch.stack((depth.sin(),depth.cos()),-1).flatten(-2)
    return (a+b).to(dtype)[None]


def act_weights(probability, cumulative, alive, final=False, epsilon=.01):
    stop = alive & ((cumulative+probability >= 1-epsilon) | final)
    weight = torch.where(stop,1-cumulative,probability)*alive.to(probability.dtype)
    remainder = torch.where(stop,1-cumulative,torch.zeros_like(probability))
    return weight, cumulative+weight, alive & ~stop, remainder


def exit_distribution(logits):
    """Hazards at all but the final loop; final loop receives the remainder."""
    survival=torch.ones_like(logits[0])
    masses=[]
    for gate in logits:
        hazard=gate.sigmoid()
        masses.append(survival*hazard)
        survival=survival*(1-hazard)
    return torch.stack((*masses,survival),-1)


def ouro_gate_loss(gates, token_losses, slope=50., threshold=.005):
    """At exit t, use the detached improvement from t to t+1 (not t-1)."""
    losses=torch.stack(token_losses,-1).detach()
    improvement=(losses[...,:-1]-losses[...,1:]).clamp_min(0)
    continuation=torch.sigmoid(slope*(improvement-threshold))
    logits=torch.stack(gates,-1)
    return F.binary_cross_entropy_with_logits(logits,1-continuation,reduction="none")


def huginn_depth(config, step, *, device="cpu"):
    """Clipped Poisson-lognormal sampled once per optimizer step.

    A step-keyed private RNG makes resume and microbatch splitting identical.
    This synchronized batch-depth policy is an explicit native adaptation.
    """
    g=torch.Generator(device="cpu").manual_seed(config.routing_seed+int(step))
    sigma=config.huginn_sigma
    tau=torch.randn((),generator=g)*sigma+math.log(config.huginn_mean-1)-sigma*sigma/2
    raw=int(torch.poisson(tau.exp(),generator=g))+1
    return min(raw,config.max_loops)


class SandwichBlock(nn.Module):
    """Huginn's block: x=norm(attn(norm(x))+x); x=norm(mlp(norm(x))+x).

    Official recurrent-pretraining `SandwichBlock` (norm_1/norm_2 around attention, norm_3/norm_4
    around the MLP). Same graph as DeepLoopBlock with alpha=1, but no beta init and used for every
    prelude / core / coda block, as in the official model.
    """
    def __init__(self, block):
        super().__init__()
        self.attn=block.attn
        self.mlp=block.mlp

    def forward(self,x,ve,cos_sin,window_size,kv_cache):
        x=norm(x+self.attn(norm(x),ve,cos_sin,window_size,kv_cache))
        return norm(x+self.mlp(norm(x)))


class DeepLoopBlock(nn.Module):
    def __init__(self, block, effective_depth):
        super().__init__()
        self.attn=block.attn
        self.mlp=block.mlp
        self.alpha=(2*effective_depth)**.5
        self.beta=(8*effective_depth)**-.5

    def forward(self,x,ve,cos_sin,window_size,kv_cache):
        x=norm(self.alpha*x+self.attn(norm(x),ve,cos_sin,window_size,kv_cache))
        return norm(self.alpha*x+self.mlp(norm(x)))
