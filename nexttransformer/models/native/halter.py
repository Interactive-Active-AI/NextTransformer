"""Causal sparse Circle exits conditioned on state change and unrolled depth."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.models.backbone import norm


class DeltaDepthHalter(nn.Module):
    def __init__(self, width, exits, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, width))
        self.depth_bias = nn.Parameter(torch.empty(exits))
        self.delta_scale = nn.Parameter(torch.empty(()))
        self.eps = eps

    @torch.no_grad()
    def init_weights(self, conditional_survival):
        nn.init.normal_(self.weight, std=self.weight.shape[-1]**-.5)
        probabilities = torch.tensor(conditional_survival, dtype=torch.float32).clamp(1e-4,1-1e-4)
        biases = -torch.distributions.Normal(0.,1.).icdf(probabilities)
        self.depth_bias.copy_(biases.to(self.depth_bias))
        nn.init.zeros_(self.delta_scale)

    def relative_change(self, before, after):
        # vector_norm has a finite zero subgradient at zero, unlike sqrt(mean(x²)).
        # All reductions are token-local and FP32, including under BF16 autocast.
        before, after = before.float(), after.float()
        scale = math.sqrt(before.shape[-1])
        delta = torch.linalg.vector_norm(after-before, dim=-1)/scale
        reference = torch.linalg.vector_norm(before, dim=-1)/scale
        return torch.log(self.eps + delta/(self.eps+reference))

    def forward(self, before, after, exit_index):
        with torch.autocast(device_type=after.device.type, enabled=False):
            base = F.linear(norm(after.float()), self.weight).squeeze(-1)
            base = base + self.depth_bias[exit_index]
            return base + self.delta_scale*self.relative_change(before, after)


def feature_forward_ops(config):
    """Approximate extra elementwise/reduction ops, separate from matrix FLOPs.

    Per exit: one d-vector subtraction, two RMS reductions, ratio/log/scale/add.
    Count sqrt/log as one operation each; this is not a hardware latency model
    or an estimate of backward cost. The upstream matrix convention omits these.
    """
    if config.architecture != 'circle':
        return 0
    from nexttransformer.models.native.depth import depth_reference
    _,survival = depth_reference(config.core_layers,config.circle_max_steps,config.circle_target_steps)
    return sum(survival[config.core_layers-1:-1])*(5*config.width+10)
