"""Registered architectures: build a model from a configuration, or load a trained checkpoint.

Two configuration types exist. `ModelConfig` covers every registered method
(baseline, looped, sparse, linear, residual, hybrid); `NativeModelConfig` covers
the historical native track (SSSL Dense, MoR, LoopFormer, Circle). Checkpoints
store tensors and a plain-dict configuration, so `load_model` rebuilds the model
from its saved configuration and loads it strictly.
"""
from dataclasses import fields
from pathlib import Path

from nexttransformer.models.registry import FAMILIES, FAMILY, METHODS

__all__ = ["FAMILIES", "FAMILY", "METHODS", "build_model", "config_from_dict", "load_model"]


def config_from_dict(raw):
    """Rebuild the configuration saved in a checkpoint or configuration file."""
    from nexttransformer.models.config import ModelConfig
    from nexttransformer.models.native.config import NativeModelConfig
    raw = dict(raw)
    cls = ModelConfig if "method" in raw else NativeModelConfig
    for field in fields(cls):  # JSON stores tuples as lists
        if isinstance(raw.get(field.name), list):
            raw[field.name] = tuple(raw[field.name])
    return cls(**raw)


def build_model(config, device="cpu", seed=42):
    """Initialize a model for `config` on `device`."""
    import torch
    from nexttransformer.models.config import ModelConfig
    from nexttransformer.models.native.config import NativeModelConfig
    if isinstance(config, ModelConfig):
        from nexttransformer.models.lm import build_model as build
        return build(config, device, seed=seed)
    if isinstance(config, NativeModelConfig):
        from nexttransformer.models.native.model import build_model as build
        return build(config, torch.device(device))
    raise TypeError(f"unsupported configuration type: {type(config).__name__}")


def load_model(checkpoint, device="cpu"):
    """Load `checkpoint_<step>/` strictly after checking its manifest; return (model, saved)."""
    from nexttransformer.runtime.checkpoint import load_checkpoint
    import torch
    checkpoint = Path(checkpoint)
    raw = torch.load(checkpoint/"model.pt", map_location="cpu", weights_only=False)["model_config"]
    model = build_model(config_from_dict(raw), device)
    saved = load_checkpoint(checkpoint, model)
    if hasattr(model, "set_step"):
        model.set_step(saved["state"]["step"])
    return model, saved
