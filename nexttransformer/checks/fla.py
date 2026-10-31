"""GPU numerics of the `fla_chunk64` kernels against the registered chunk algorithm.

For DeltaNet, KDA, RetNet, GDN and ELU+1 at the formal head shape [B, T, 12, 128], the FLA
Triton path (BF16 operands, as in training) and the plain-torch chunk form in
FP32 are compared with an FP64 oracle (the torch chunk form, which matches the
recurrent scan on CPU). Outputs and gradients of every raw input, including KDA/GDN
gates and beta, are compared by relative L2 error. Tolerances are fixed before
running: outputs <= 1e-2, gradients <= 3e-2. A Hopper kernel defect such as FLA
issue #640 (wrong dk at BK=64) shows up as an order-of-magnitude larger error.
Bounded check only; it never trains a model.

With --checkpoint, the inputs are instead the actual mixer inputs (q, k, v, log
decay, beta after the feature maps) of every layer of a trained checkpoint on
one real validation row, captured while the model runs its training backend.
The oracle is then the FP64 recurrent scan: trained KDA gates can make the
within-chunk cumulative log decay exceed the exp range of the torch chunk form
(exp(-b) overflows below about -88 in FP32), which FLA avoids with sub-chunk
reference points. Gate and ELU+1 denominator statistics are recorded with it.

ELU checkpoints instead check the complete production path from raw BF16
projections through features, attention, normalization/gain and the final cast.
The VJP uses the actual LM-loss upstream gradient, not a random probe. The FP64
oracle is the independent chunk_reference (also checked against the scan).
No optimizer updates are performed. --split train restores rank 0's saved
data cursor; val uses the fixed first validation row.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
import torch
from torch.nn import functional as F

from nexttransformer.models import linear
from nexttransformer.models.linear import chunk_attention, recurrent_attention
from nexttransformer.checks.validate import source_hash

TOLERANCE = dict(output=1e-2, gradient=3e-2)
ELU_KINDS = ("elu_linear", "elu_norm", "elu_qknorm")
HEADS, DIM, CHUNK = 12, 128, 64


def raw_inputs(kind, batch, length, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    def normal(*shape): return torch.randn(*shape, generator=g, device=device, dtype=torch.float64)
    raw = dict(q=normal(batch, length, HEADS, DIM), k=normal(batch, length, HEADS, DIM), v=normal(batch, length, HEADS, DIM))
    if kind == "deltanet": raw.update(logit=normal(batch, length, HEADS))
    if kind == "kda": raw.update(gate=normal(batch, length, HEADS, DIM), logit=normal(batch, length, HEADS))
    if kind == "gdn": raw.update(gate=normal(batch, length, HEADS), logit=normal(batch, length, HEADS))
    return raw, normal(batch, length, HEADS, DIM)


def gdn_gate_parameters(device, dtype):
    """Per-head A and dt_bias spanning GDN's initialization ranges (A in (0, 16], dt in [1e-3, 1e-1])."""
    a = torch.linspace(1., linear.GDN_A_RANGE[1], HEADS, device=device, dtype=dtype)
    dt = torch.logspace(math.log10(linear.GDN_DT_RANGE[0]), math.log10(linear.GDN_DT_RANGE[1]), HEADS,
                        device=device, dtype=dtype)
    return a, dt+torch.log(-torch.expm1(-dt))


def qk_normalize(z):
    """elu_qknorm's q/k scale: the model's FP32 RMSNorm (FP32 epsilon) times the temperature."""
    z = z if z.dtype == torch.float64 else z.float()
    return F.rms_norm(z, (z.shape[-1],), eps=torch.finfo(torch.float32).eps)*linear.ELU_QK_TEMPERATURE


def forward(kind, raw, dtype, implementation, gate_scale=1.):
    """Feature maps exactly as LinearMixer at initialization (KDA: log_decay=0, dt_bias=-3;
    GDN: the per-head A/dt spread of `gdn_gate_parameters`). gate_scale multiplies GDN's
    log decay to stress strong forgetting; its pairwise chunk oracle has no positive exponent."""
    if gate_scale != 1. and kind != "gdn":
        raise ValueError("--gate-scale is a GDN stress option (the KDA chunk oracle overflows)")
    leaves = {name: x.detach().to(dtype).requires_grad_() for name, x in raw.items()}
    q, k, v = leaves["q"], leaves["k"], leaves["v"]
    extra = {}
    if kind == "deltanet":
        q, k = F.normalize(q, dim=-1)*DIM**-.5, F.normalize(k, dim=-1)
        extra = dict(beta=torch.sigmoid(leaves["logit"]))
    elif kind in ("kda", "gdn"):
        gate = leaves["gate"] if dtype == torch.float64 else leaves["gate"].float()
        if kind == "kda":
            q, k = F.normalize(q, dim=-1)*DIM**-.5, F.normalize(k, dim=-1)
            extra = dict(g=-F.softplus(gate-3.))
        else:  # raw q/k: the production path normalizes them (inside the FLA kernel on CUDA)
            a, dt_bias = gdn_gate_parameters(q.device, gate.dtype)
            extra = dict(g=-a*F.softplus(gate+dt_bias)*gate_scale, qk_l2norm_scale=DIM**-.5)
        extra["beta"] = torch.sigmoid(leaves["logit"])
    elif kind == "retnet":
        gamma = 1-2.**(-5-torch.arange(HEADS, device=q.device, dtype=torch.float64 if dtype == torch.float64 else torch.float32))
        extra = dict(g=gamma.log())
    else:
        if kind == "elu_qknorm": q, k = qk_normalize(q), qk_normalize(k)
        q, k = linear.elu_feature_map(q), linear.elu_feature_map(k)
    y = chunk_attention(q, k, v, kind=kind, chunk=CHUNK, implementation=implementation, **extra)
    if kind in ("elu_norm", "elu_qknorm"):
        y = F.rms_norm(y, (DIM,), eps=torch.finfo(torch.bfloat16).eps)
    if kind in ELU_KINDS: y = y.to(dtype)
    return leaves, y


def relative(a, b):
    a, b = a.double(), b.double()
    return float((a-b).norm()/b.norm().clamp_min(1e-300))


def check(kind, batch, length, seed, device, gate_scale=1.):
    raw, weight = raw_inputs(kind, batch, length, seed, device)
    results = {}
    for label, dtype, implementation in (("oracle_fp64", torch.float64, "torch"),
                                         ("torch_chunk_fp32", torch.float32, "torch"),
                                         ("fla_bf16", torch.bfloat16, "fla")):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        leaves, y = forward(kind, raw, dtype, implementation, gate_scale)
        grads = torch.autograd.grad((y.double()*weight).sum(), list(leaves.values()))
        torch.cuda.synchronize()
        results[label] = dict(output=y.detach().double(), grads=dict(zip(leaves, (g.double() for g in grads))),
                              output_dtype=str(y.dtype), peak_bytes=torch.cuda.max_memory_allocated())
    oracle = results.pop("oracle_fp64")
    report = {}
    for label, item in results.items():
        errors = dict(output=relative(item["output"], oracle["output"]),
                      **{"grad_"+name: relative(g, oracle["grads"][name]) for name, g in item["grads"].items()})
        passed = errors["output"] <= TOLERANCE["output"] and all(
            v <= TOLERANCE["gradient"] for name, v in errors.items() if name.startswith("grad_"))
        report[label] = dict(errors=errors, passed=passed, output_dtype=item["output_dtype"], peak_bytes=item["peak_bytes"])
    return report


def captured_inputs(config_path, checkpoint, resources, data_dir, device):
    """Every layer's chunk_attention inputs for one validation row of a checkpoint."""
    from nexttransformer.runtime.data import BestFitStream
    from nanochat.tokenizer import RustBPETokenizer
    from nexttransformer.protocol.spec import load_config
    from nexttransformer.models.lm import build_model
    config = load_config(config_path)
    model = build_model(config, device)
    state = torch.load(Path(checkpoint)/"model.pt", map_location=device, weights_only=False)["model"]
    model.load_state_dict(state)
    model.eval()
    tokenizer = RustBPETokenizer.from_directory(Path(resources)/"tokenizer")
    x, y = BestFitStream(data_dir, tokenizer, "val", 0, 1, config.sequence_len).batch(1, device)
    layers, original = [], linear.chunk_attention
    def capture(q, k, v, *, kind, g=None, beta=None, chunk, implementation=None, qk_l2norm_scale=None):
        layers.append(dict(kind=kind, q=q.detach(), k=k.detach(), v=v.detach(), qk_l2norm_scale=qk_l2norm_scale,
                           g=None if g is None else g.detach(), beta=None if beta is None else beta.detach()))
        return original(q, k, v, kind=kind, g=g, beta=beta, chunk=chunk, implementation=implementation,
                        qk_l2norm_scale=qk_l2norm_scale)
    linear.chunk_attention = capture
    try:
        with torch.no_grad(): loss = float(model(x, y))
    finally:
        linear.chunk_attention = original
    return config, layers, loss


def input_statistics(item):
    """Trained gate range (KDA, GDN), per-head decay (RetNet) and ELU+1 denominator."""
    def quantiles(x, qs=(0.0, 0.001, 0.01, 0.5)):
        x = x.flatten().float()
        x = x[torch.randperm(x.numel(), device=x.device)[:2**24]] if x.numel() > 2**24 else x
        return {f"q{q:g}": float(x.quantile(q)) for q in qs}
    kind, stats = item["kind"], {}
    if kind in ("kda", "gdn"):
        g, beta = item["g"].float(), item["beta"].float()
        # Chunk position is dim -2 for KDA's per-feature gate, -1 for GDN's per-head gate.
        within = linear._to_chunks(g, CHUNK).cumsum(-2 if kind == "kda" else -1)
        stats = dict(log_decay=quantiles(g), log_decay_below_minus5=float((g < -5).float().mean()),
                     decay_below_0p5=float((g < math.log(.5)).float().mean()),
                     decay_above_0p999=float((g > math.log(.999)).float().mean()),
                     within_chunk_cumulative_min=float(within.min()),
                     within_chunk_beyond_fp32_exp=float((within < -88.).float().mean()),
                     beta=dict(mean=float(beta.mean()), min=float(beta.min()), max=float(beta.max())),
                     beta_below_0p01=float((beta < .01).float().mean()), beta_above_0p99=float((beta > .99).float().mean()))
    elif kind in ELU_KINDS:  # elu_norm/elu_qknorm: phi growth only (no division in the model)
        q, k = item["q"].float(), item["k"].float()
        denominator = (q*k.cumsum(1)).sum(-1)
        per_position = denominator/torch.arange(1, q.shape[1]+1, device=q.device)[None, :, None]
        stats = dict(denominator=quantiles(denominator), denominator_per_position=quantiles(per_position))
    return stats


def check_trained(item, seed):
    """FLA (training dtypes) vs FP64 recurrent oracle on one layer's captured inputs.
    GDN's captured q/k are raw: the oracle normalizes them, the chunk paths take the scale."""
    kind, scale = item["kind"], item.get("qk_l2norm_scale")
    names = [n for n in ("q", "k", "v", "g", "beta") if item[n] is not None and not (kind == "retnet" and n == "g")]
    weight = torch.randn(item["v"].shape, generator=torch.Generator(device=item["v"].device).manual_seed(seed),
                         device=item["v"].device, dtype=torch.float64)
    results = {}
    for label in ("oracle_fp64_recurrent", "torch_chunk_fp32", "fla_bf16"):
        leaves = {n: (item[n].double() if label.startswith("oracle") else
                      item[n].float() if label.startswith("torch") else item[n]).clone().requires_grad_() for n in names}
        q, k, v = leaves["q"], leaves["k"], leaves["v"]
        g, beta = leaves.get("g", item["g"]), leaves.get("beta")
        if label.startswith("oracle"):
            if kind == "retnet": decay = item["g"].double().exp()[None, None, :, None].expand(*q.shape[:3], 1)
            elif kind == "gdn": decay = g.exp().unsqueeze(-1)
            else: decay = None if g is None else g.exp()
            if scale is not None: q, k = F.normalize(q, dim=-1)*scale, F.normalize(k, dim=-1)
            y, _ = recurrent_attention(q, k, v, kind=kind, decay=decay, beta=beta)
        else:
            y = chunk_attention(q, k, v, kind=kind, g=g, beta=beta, chunk=CHUNK, qk_l2norm_scale=scale,
                                implementation="torch" if label.startswith("torch") else "fla")
        grads = torch.autograd.grad((y.double()*weight).sum(), list(leaves.values()))
        results[label] = dict(output=y.detach().double(), grads=dict(zip(leaves, (x.double() for x in grads))),
                              finite=bool(torch.isfinite(y).all()))
    oracle = results.pop("oracle_fp64_recurrent")
    report = {}
    for label, item_result in results.items():
        errors = dict(output=relative(item_result["output"], oracle["output"]),
                      **{"grad_"+n: relative(x, oracle["grads"][n]) for n, x in item_result["grads"].items()})
        passed = all(math.isfinite(e) for e in errors.values()) and errors["output"] <= TOLERANCE["output"] and all(
            e <= TOLERANCE["gradient"] for n, e in errors.items() if n.startswith("grad_"))
        report[label] = dict(errors=errors, passed=passed, finite_output=item_result["finite"])
    return report


def check_elu_layer(item, kind, gain=None):
    """Compare a real production VJP to FP64, starting before feature mapping."""
    names = ["q", "k", "v"]
    leaves = [item[n].detach().double().requires_grad_() for n in names]
    q, k, v = leaves
    # Independent evaluation; do not call the production feature helper here.
    if kind == "elu_qknorm": q, k = qk_normalize(q), qk_normalize(k)
    q, k = (torch.where(z > 0, z+1, z.clamp_max(0).exp()) for z in (q, k))
    y = linear.chunk_reference(q, k, v, kind=kind, chunk=CHUNK)
    if kind == "elu_linear":
        y = y/((q*k.cumsum(1)).sum(-1, keepdim=True)+1e-6)
    else:
        effective_gain = gain.detach().to(item["output"].dtype).double().requires_grad_()
        leaves.append(effective_gain); names.append("gain")
        y = F.rms_norm(y, (v.shape[-1],), eps=torch.finfo(item["output"].dtype).eps)*effective_gain
    gradients = torch.autograd.grad(y, leaves, item["output"].grad.double())
    errors = dict(output=relative(item["output"], y))
    for name, expected in zip(names, gradients):
        actual = gain.grad if name == "gain" else item[name].grad
        errors["grad_"+name] = relative(actual, expected)
    passed = all(math.isfinite(e) and e <= TOLERANCE["output" if n == "output" else "gradient"]
                 for n, e in errors.items())
    return dict(errors=errors, passed=passed)


def check_elu_checkpoint(config, checkpoint, resources, data_dir, device, split="val"):
    """Capture raw projection gradients and actual LM upstream on one real row."""
    from nexttransformer.runtime.data import BestFitStream
    from nanochat.tokenizer import RustBPETokenizer
    from nexttransformer.models.lm import build_model
    model = build_model(config, device)
    state = torch.load(Path(checkpoint)/"model.pt", map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(state["model"])
    del state
    model.train()
    tokenizer = RustBPETokenizer.from_directory(Path(resources)/"tokenizer")
    stream = BestFitStream(data_dir, tokenizer, split, 0, 8 if split == "train" else 1, config.sequence_len)
    if split == "train":
        saved = torch.load(Path(checkpoint)/"rank_0000.pt", map_location="cpu", weights_only=False)
        stream.load_state_dict(saved["data"])
        del saved
    x, targets = stream.batch(1, device)
    layers = [dict() for _ in model.transformer.h]
    handles, original = [], linear.chunk_attention
    for i, block in enumerate(model.transformer.h):
        for name in ("q", "k"):
            def projection(module, args, output, i=i, name=name):
                output.retain_grad()
                layers[i][name] = output
            handles.append(getattr(block.attn, "c_"+name).register_forward_hook(projection))
        def before_projection(module, args, i=i):
            args[0].retain_grad()
            layers[i]["output"] = args[0]
        handles.append(block.attn.c_proj.register_forward_pre_hook(before_projection))
    index = 0
    def capture(q, k, v, **kwargs):
        nonlocal index
        v.retain_grad(); layers[index]["v"] = v
        index += 1
        return original(q, k, v, **kwargs)
    linear.chunk_attention = capture
    try:
        loss = model(x, targets); loss.backward()
    finally:
        linear.chunk_attention = original
        for handle in handles: handle.remove()
    if index != len(layers): raise RuntimeError("ELU check requires every layer to use the chunk backend")
    rows = []
    for i, (item, block) in enumerate(zip(layers, model.transformer.h)):
        # Retained gradients belong to the production tensors, not new views.
        shaped = {}
        for name, value in item.items():
            shaped[name] = value.detach().reshape(1, config.sequence_len, config.heads, config.head_dim)
            shaped[name].grad = value.grad.reshape_as(shaped[name])
        stats = {name: dict(min=float(item[name].min()), max=float(item[name].max()),
                           legacy_false_zero_fraction=float(((F.elu(item[name].detach())+1) == 0).float().mean()))
                 for name in ("q", "k")}
        if config.method == "elu_qknorm":
            for name in ("q", "k"):
                pre = qk_normalize(item[name].detach().reshape(1, config.sequence_len, config.heads, config.head_dim))
                stats[name].update(normalized_min=float(pre.min()), normalized_max=float(pre.max()))
        report = check_elu_layer(shaped, config.method, getattr(block.attn, "output_gain", None))
        row = dict(kind=config.method, layer=i, length=config.sequence_len, statistics=stats, full_path=report)
        rows.append(row); print(json.dumps(row), flush=True)
    return rows, dict(checkpoint=str(checkpoint), method=config.method, row_loss=float(loss.detach()),
                      split=split, rank=0, batch=1, layers=len(layers), optimizer_updates=0,
                      oracle="fp64 independent chunk_reference; raw projections through output cast",
                      upstream="actual LM-loss gradient at c_proj input",
                      criterion="full_path output <= 1%; all q/k/v/gain gradients <= 3% on every layer")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kinds", nargs="+", default=["deltanet", "kda", "retnet", "elu_linear", "elu_norm", "elu_qknorm", "gdn"])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lengths", nargs="+", type=int, default=[2048, 1000])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--gate-scale", type=float, default=1., help="GDN only: multiply the log decay (stress)")
    p.add_argument("--checkpoint", help="trained checkpoint directory: check captured layer inputs instead")
    p.add_argument("--config", help="admitted config of --checkpoint")
    p.add_argument("--resources")
    p.add_argument("--data-dir")
    p.add_argument("--split", choices=("val", "train"), default="val",
                   help="ELU checkpoint check: fixed val prefix or saved rank-0 train cursor")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    device = torch.device("cuda")
    import fla, triton
    from fla.utils import check_shared_mem
    rows, extra = [], {}
    if args.checkpoint:
        from nexttransformer.protocol.spec import load_config
        config = load_config(args.config)
        if config.method in ELU_KINDS:
            rows, extra = check_elu_checkpoint(config, args.checkpoint, args.resources, args.data_dir, device, args.split)
            extra["config"] = args.config
        else:
            if args.split != "val": raise ValueError("--split train is supported for ELU checkpoint checks")
            config, layers, loss = captured_inputs(args.config, args.checkpoint, args.resources, args.data_dir, device)
            # A hybrid's attention layers make no chunk call: report physical layer numbers.
            from nexttransformer.models.config import HYBRID, LINEAR
            physical = ([i for i, kind in enumerate(config.layer_mixers) if kind in LINEAR]
                        if config.method in HYBRID else list(range(len(layers))))
            if len(physical) != len(layers): raise RuntimeError("captured chunk calls differ from the linear layers")
            extra = dict(checkpoint=str(args.checkpoint), config=args.config, method=config.method,
                         validation_row_loss=loss, layers=len(layers), oracle="fp64 recurrent scan",
                         criterion="fla_bf16 within tolerance on every layer (torch_chunk_fp32 informational)")
            for index, item in enumerate(layers):
                rows.append(dict(kind=item["kind"], layer=physical[index], length=item["q"].shape[1],
                                 statistics=input_statistics(item), **check_trained(item, index)))
                print(json.dumps(rows[-1]), flush=True)
    else:
        for kind in args.kinds:
            for length in args.lengths:
                for seed in args.seeds:
                    report = check(kind, args.batch, length, seed, device, args.gate_scale)
                    rows.append(dict(kind=kind, batch=args.batch, length=length, seed=seed,
                                     gate_scale=args.gate_scale, **report))
                    print(json.dumps(rows[-1]), flush=True)
    passed = all(r["full_path" if "full_path" in r else "fla_bf16"]["passed"] for r in rows)
    record = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status="passed" if passed else "failed",
                  tolerance=TOLERANCE, device=torch.cuda.get_device_name(device),
                  capability=list(torch.cuda.get_device_capability(device)),
                  hopper_shared_memory=bool(check_shared_mem("hopper")), torch=torch.__version__,
                  triton=triton.__version__, fla_core=importlib.metadata.version("fla-core"),
                  backend_dispatch_disabled=os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1",
                  source_hash=source_hash(), **extra, rows=rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2)+"\n")
    if not passed: raise SystemExit("FLA numerics exceed the registered tolerance")


if __name__ == "__main__":
    main()
