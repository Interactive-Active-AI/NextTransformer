"""Test plugin for the ffn hook: a gated linear unit with the shell's FFN width."""
from torch import nn
from torch.nn import functional as F

from nexttransformer.models.backbone import Linear


class GLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.width, 2 * config.mlp_width, bias=False)
        self.c_proj = Linear(config.mlp_width, config.width, bias=False)

    def forward(self, x):
        value, gate = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(F.silu(gate) * value)


def ffn(config, layer):
    return GLU(config)


def check(config):
    if config.option("activation") != "silu":
        raise ValueError("toy_glu implements the silu activation only")
