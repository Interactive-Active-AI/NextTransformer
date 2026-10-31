"""Test plugin for the mixer hook: causal attention whose output passes a learned sigmoid gate."""
import torch
from torch import nn

from nexttransformer.models.backbone import CausalSelfAttention, Linear


class GatedAttention(CausalSelfAttention):
    def __init__(self, config, layer):
        super().__init__(config, layer)
        self.gate = Linear(config.width, config.width, bias=False)
        self.bias = config.option("gate_bias")

    def forward(self, x, ve, cos_sin, window, kv_cache):
        return super().forward(x, ve, cos_sin, window, kv_cache) * torch.sigmoid(self.gate(x) + self.bias)


def mixer(config, layer):
    return GatedAttention(config, layer)


def init(model):
    for block in model.transformer.h:
        nn.init.zeros_(block.attn.gate.weight)
