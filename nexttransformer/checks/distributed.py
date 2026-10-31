"""One bounded eight-rank optimizer update per method, never a training run.

Default uses small architecture dimensions but REAL 2048-token rows and REAL
1,048,576-token global batch. --full-dimensions requires calibrated configs.
Synthetic inputs are labelled and never reported as ClimbMix experiments.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nanochat.common import COMPUTE_DTYPE

from nexttransformer.models.native.config import NativeTrainConfig
from nexttransformer.models.config import METHODS, debug_config
from nexttransformer.protocol.spec import config_file, Protocol, load_config
from nexttransformer.models.lm import build_model
from nexttransformer.models.sparse import BACKENDS as SPARSE_BACKENDS
from nexttransformer.training.optimizer import build_optimizer,set_schedule
from nexttransformer.checks.validate import source_hash


def digest(model):
    result=hashlib.sha256()
    for name,p in model.named_parameters():
        result.update(name.encode());result.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--micro-batch",type=int,default=64)
    parser.add_argument("--full-dimensions",action="store_true")
    parser.add_argument("--config-dir",type=Path,default=Path("configs"))
    parser.add_argument("--stage",choices=("joint","gate_only","burnin"),default="joint")
    parser.add_argument("--sparse-backend",choices=SPARSE_BACKENDS,default="reference")
    parser.add_argument("--smoke",action="store_true",help="bounded synthetic check on other rank counts; keeps the full global batch")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    initial_source=source_hash()
    p=Protocol();rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"])
    local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);device=torch.device("cuda",local)
    if args.smoke:
        denominator=world*args.micro_batch*p.sequence_len
        if denominator<=0 or p.global_batch_tokens%denominator:
            raise ValueError("smoke microbatch/ranks must divide the common global batch")
        accumulation=p.global_batch_tokens//denominator
    else:accumulation=p.accumulation(args.micro_batch,world)
    dist.init_process_group("nccl",device_id=device);torch.set_num_threads(2)
    results=[]
    for method in args.methods:
        c=load_config(config_file(args.config_dir,method)) if args.full_dimensions else debug_config(method,sequence_len=p.sequence_len)
        m=build_model(c,device)
        m.set_sparse_backend(args.sparse_backend)
        step=p.steps-c.ouro_gate_steps if args.stage=="gate_only" else 0 if args.stage=="burnin" else c.burnin_steps+1
        if args.stage=="gate_only" and method!="ouro":raise ValueError("gate-only stage belongs to Ouro")
        stage=m.set_training_stage(step)
        wrapper=DDP(m,device_ids=[local],broadcast_buffers=False)
        opt=build_optimizer(m);set_schedule(opt,step,NativeTrainConfig())
        before=digest(m);torch.cuda.reset_peak_memory_stats();dist.barrier();start=time.monotonic()
        rows_per_rank=p.global_batch_tokens//p.sequence_len//world
        tokens=0;loss_sum=torch.zeros((),device=device)
        for micro in range(accumulation):
            offset=step*512+rank*rows_per_rank+micro*args.micro_batch
            m.set_row_offset(offset)
            # Stateless token rows, invariant to microbatch splitting.
            positions=torch.arange(args.micro_batch*(p.sequence_len+1),device=device).reshape(args.micro_batch,-1)
            row_ids=torch.arange(offset,offset+args.micro_batch,device=device)[:,None]
            data=(positions.remainder(p.sequence_len+1)+row_ids*17+42).remainder(c.vocab_size)
            x,y=data[:,:-1].contiguous(),data[:,1:].contiguous()
            context=wrapper.no_sync() if micro<accumulation-1 else nullcontext()
            with context:
                loss=wrapper(x,y)/accumulation;loss.backward()
            loss_sum+=loss.detach();tokens+=x.numel()*world
        if tokens!=p.global_batch_tokens:raise RuntimeError("global token batch drift")
        for name,param in m.named_parameters():
            if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()):
                raise RuntimeError("missing or nonfinite gradient: "+name)
        opt.step();torch.cuda.synchronize();elapsed=time.monotonic()-start
        after=digest(m);hashes=[None]*world;dist.all_gather_object(hashes,after)
        if len(set(hashes))!=1 or before==after:raise RuntimeError("optimizer did not synchronize a real parameter update")
        dist.all_reduce(loss_sum);loss_mean=float(loss_sum/world)
        result=dict(method=method,status="passed",stage=args.stage,trainable_stage=stage,
                    sparse_backend=args.sparse_backend,model_hash=c.fingerprint,
                    full_dimensions=args.full_dimensions,input_shape=[args.micro_batch,p.sequence_len],
                    global_tokens=tokens,world_size=world,accumulation=accumulation,
                    optimizer_updates=1,loss=loss_mean,seconds=elapsed,all_rank_parameters_equal=True,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),synthetic_data=True,smoke=args.smoke)
        results.append(result)
        if rank==0:
            if source_hash()!=initial_source:raise RuntimeError("reference sources changed during validation")
            print(json.dumps(result),flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),
                source_hash=initial_source,protocol_hash=p.fingerprint,torch=torch.__version__,
                dtype=str(COMPUTE_DTYPE),device=torch.cuda.get_device_name(),seed=42,results=results),indent=2)+"\n")
        del opt,wrapper,m,loss,x,y,data
        gc.collect();torch.cuda.empty_cache();dist.barrier()
    dist.destroy_process_group()


if __name__=="__main__":main()
