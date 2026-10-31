"""Bounded, machine-readable architecture checks; never starts a 50B run.

Run with python or torchrun. In smoke/target modes ranks test disjoint models;
These modes do not perform a global optimizer update; use distributed_check
for the independent eight-way DDP accumulation and optimizer validation.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist
from nanochat.common import COMPUTE_DTYPE

from nexttransformer.models.config import METHODS, candidate_config, debug_config
from nexttransformer.protocol.spec import config_file, Protocol, load_config
from nexttransformer.models.lm import build_model
from nexttransformer.models.sparse import BACKENDS as SPARSE_BACKENDS


def source_hash():
    root=Path(__file__).resolve().parents[2]
    digest=hashlib.sha256()
    files=[p for folder in ("nexttransformer", "nanochat")
           for p in (root/folder).rglob("*.py")
           if not {"results", "tests", "__pycache__", ".agent-work"}.intersection(p.relative_to(root).parts)]
    for p in sorted(files):
        digest.update(p.relative_to(root).as_posix().encode());digest.update(p.read_bytes())
    return digest.hexdigest()


def activate(model):
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.02)
            block.mlp.c_proj.weight.normal_(std=.02)


def check_one(method,args,device):
    if args.config_dir:c=load_config(config_file(args.config_dir,method))
    else:c=candidate_config(method) if args.mode=="target" else debug_config(method)
    model=build_model(c,device);activate(model)
    model.set_step(0 if args.stage=="burnin" else c.burnin_steps+1)
    model.set_sparse_backend(args.sparse_backend)
    t=args.sequence_length or (64 if args.mode=="target" else c.sequence_len)
    if t>c.sequence_len:raise ValueError("test length exceeds the model context")
    x=torch.randint(c.vocab_size,(args.micro_batch,t),device=device)
    y=torch.randint(c.vocab_size,x.shape,device=device)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    started=time.monotonic()
    loss=model(x,y);loss.backward()
    if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
    missing=[]
    for name,p in model.named_parameters():
        if p.requires_grad and p.grad is None:missing.append(name)
        if p.grad is not None and not torch.isfinite(p.grad).all():raise RuntimeError("nonfinite gradient: "+name)
    if missing:raise RuntimeError("missing gradients: "+", ".join(missing))
    model.zero_grad(set_to_none=True);model.eval()
    with torch.no_grad():
        z=x.clone();z[:,t//2:]=(z[:,t//2:]+7)%c.vocab_size
        reference=model(x);perturbed=model(z)
        error=float((reference[:,:t//2]-perturbed[:,:t//2]).abs().max())
        if error>3e-3:raise RuntimeError(f"causal prefix changed: {error}")
    if device.type=="cuda":torch.cuda.synchronize()
    result=dict(method=method,status="passed",mode=args.mode,
        model_hash=c.fingerprint,protocol_hash=Protocol().fingerprint,
        configured_context=c.sequence_len,test_shape=list(x.shape),width=c.width,
        physical_layers=c.n_layer,parameters=sum(p.numel() for p in model.parameters()),
        loss=float(loss.detach()),causal_max_abs=error,elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else None,
        scope=("full-config-full-context" if t==c.sequence_len else "full-config-short-sequence")
              if args.mode=="target" else "reference-smoke",
        full_global_update=False,stage=args.stage,sparse_backend=args.sparse_backend)
    del model,loss,x,y,z,reference,perturbed
    gc.collect()
    if device.type=="cuda":torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("smoke","target"),default="smoke")
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--sequence-length",type=int)
    p.add_argument("--micro-batch",type=int,default=1)
    p.add_argument("--config-dir")
    p.add_argument("--stage",choices=("burnin","joint"),default="joint")
    p.add_argument("--sparse-backend",choices=SPARSE_BACKENDS,default="reference")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    initial_source=source_hash()
    rank=int(os.environ.get("RANK",0));world=int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0))
    if args.device=="cuda":torch.cuda.set_device(local)
    device=torch.device(args.device,local) if args.device=="cuda" else torch.device("cpu")
    torch.set_num_threads(2)
    # Independent model tests need only a CPU completion barrier. A CUDA OOM
    # must not prevent publishing the failed rank's diagnostic JSON.
    if world>1:dist.init_process_group("gloo")
    results=[]
    for method in args.methods[rank::world]:
        try:result=check_one(method,args,device)
        except Exception as e:
            result=dict(method=method,status="failed",error=str(e),traceback=traceback.format_exc())
            gc.collect()
            if device.type=="cuda":torch.cuda.empty_cache()
        results.append(result)
        print(json.dumps(dict(rank=rank,**result)),flush=True)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    shard=out.with_name(out.stem+f".rank{rank}"+out.suffix)
    shard.write_text(json.dumps(results,indent=2)+"\n")
    if world>1:dist.barrier()
    if rank==0:
        all_results=[]
        for r in range(world):all_results.extend(json.loads(out.with_name(out.stem+f".rank{r}"+out.suffix).read_text()))
        if source_hash()!=initial_source:raise RuntimeError("reference sources changed during validation")
        record=dict(created_at=datetime.now(timezone.utc).isoformat(),source_hash=initial_source,
                    torch=torch.__version__,device=torch.cuda.get_device_name() if device.type=="cuda" else "cpu",
                    dtype=str(COMPUTE_DTYPE),world_size=world,mode=args.mode,seed=42,results=all_results)
        out.write_text(json.dumps(record,indent=2)+"\n")
    if world>1:dist.destroy_process_group()
    if any(r["status"]!="passed" for r in results):raise SystemExit(1)


if __name__=="__main__":main()
