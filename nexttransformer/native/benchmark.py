"""Measure real-data training updates at fixed global batch; never saves training checkpoints."""
from nexttransformer.runtime.site import add_runtime_arguments, bootstrap_runtime
if __name__ == '__main__':
    bootstrap_runtime()

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nexttransformer.models.native.config import load_config
from nexttransformer.runtime.files import atomic_json
from nexttransformer.runtime.data import BestFitStream, verify_resources
from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.evaluation.native import evaluate_bpb
from nexttransformer.models.native.model import build_model, theoretical_training_flops
from nexttransformer.runtime.statistics import training_stats, accumulate_stats
from nexttransformer.native.optimizer import build_optimizer, set_schedule, clip_halter_gradients, clip_condition_gradients
from nexttransformer.runtime.process import initialize, autocast, synchronize, reduce_sum, source_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    add_runtime_arguments(p)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--circle-stage-step',type=int,default=10000,help='benchmark curriculum offset; does not change data indices or LR')
    p.add_argument('--stress-full-path',action='store_true')
    args=p.parse_args()
    if args.micro_batch is None or args.warmup<2 or args.steps<2:
        p.error('provide --micro-batch; warmup and measured steps must each be >= 2')
    mc,tc=load_config(args.config)
    rank,world,device=initialize(tc.seed)
    if device.type!='cuda': raise ValueError('batch tuning requires CUDA GPUs')
    accumulation=tc.accumulation(mc,world,args.micro_batch)
    record,manifest_hash=verify_resources(args.resources,args.data_dir)
    tokenizer=RustBPETokenizer.from_directory(args.resources/'tokenizer')
    data_dir=args.data_dir or args.resources/'base_data_climbmix'
    train=BestFitStream(data_dir,tokenizer,'train',rank,world,mc.sequence_len)
    model=build_model(mc,device)
    wrapped=DDP(model,device_ids=[device.index],broadcast_buffers=False) if world>1 else model
    optimizer=build_optimizer(model,tc)
    model.train()
    if args.stress_full_path:
        with torch.no_grad():
            if mc.architecture=='mor':
                for router in model.routers: router.weight.zero_();router.bias.fill_(10.)
            elif mc.architecture=='circle':
                model.halter.weight.zero_()
                model.halter.depth_bias.fill_(-10.)
                model.halter.delta_scale.zero_()
    global_sequences=tc.global_batch_tokens//mc.sequence_len
    durations=[]; optimizer_times=[]; data_times=[]; losses=[]
    torch.cuda.reset_peak_memory_stats(device)
    warmup_allocated=warmup_reserved=0
    wall_start=None
    for step in range(args.warmup+args.steps):
        model.set_circle_step(step+args.circle_stage_step)
        if step==args.warmup:
            warmup_allocated=torch.cuda.max_memory_allocated(device)
            warmup_reserved=torch.cuda.max_memory_reserved(device)
            if world>1: dist.barrier()
            synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            wall_start=time.time()
        synchronize(device); started=time.perf_counter(); data_seconds=0.0
        optimizer.zero_grad(set_to_none=True)
        set_schedule(optimizer,step,tc)
        # Same update order, statistics reduction and data stream as native_train.
        stats=training_stats(mc,device)
        for micro in range(accumulation):
            t=time.perf_counter()
            x,y=train.batch(args.micro_batch,device)
            data_seconds+=time.perf_counter()-t
            sync=wrapped.no_sync() if world>1 and micro<accumulation-1 else nullcontext()
            with sync,autocast(device):
                out=wrapped(x,y)
                (out['loss']/accumulation).backward()
            s=model.last_stats
            accumulate_stats(stats,out,s,accumulation,world,mc)
        torch.nn.utils.clip_grad_norm_(model.parameters(),tc.grad_clip or float("inf"),error_if_nonfinite=True)
        clip_halter_gradients(model)
        clip_condition_gradients(model)
        synchronize(device);t=time.perf_counter();optimizer.step();synchronize(device)
        optimizer_seconds=time.perf_counter()-t
        reduce_sum(stats)
        values=stats.tolist()
        if int(values[4])!=tc.global_batch_tokens:raise RuntimeError('token budget drift')
        model.update_route_budget(values[6:6+mc.routing_steps],values[4],step+args.circle_stage_step)
        synchronize(device);duration=time.perf_counter()-started
        if step>=args.warmup:
            durations.append(duration);optimizer_times.append(optimizer_seconds);data_times.append(data_seconds)
            losses.append(float(stats[0]))
    synchronize(device);wall_end=time.time()
    props=torch.cuda.get_device_properties(device)
    local={'rank':rank,'device':device.index,'name':props.name,'uuid':str(getattr(props,'uuid','')),
           'total_memory_bytes':props.total_memory,
           'peak_allocated_bytes':max(warmup_allocated,torch.cuda.max_memory_allocated(device)),
           'peak_reserved_bytes':max(warmup_reserved,torch.cuda.max_memory_reserved(device)),
           'steady_peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
           'steady_peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
           'step_seconds':durations,'optimizer_seconds':optimizer_times,'data_seconds':data_times,
           'measurement_start':wall_start,'measurement_end':wall_end}
    ranks=[None]*world
    if world>1:dist.all_gather_object(ranks,local)
    else:ranks=[local]
    # Exercise a complete evaluation microbatch on every rank; report separately.
    val_tokens=world*args.micro_batch*mc.sequence_len
    result=evaluate_bpb(model,tokenizer,data_dir,args.resources,val_tokens,args.micro_batch,rank,world,device)
    if rank==0:
        slowest=[max(r['step_seconds'][i] for r in ranks) for i in range(args.steps)]
        report={'model':asdict(mc),'train':asdict(tc),'world_size':world,'micro_batch':args.micro_batch,
                'accumulation':accumulation,'warmup_steps':args.warmup,'measured_steps':args.steps,
                'step_seconds_max_rank':slowest,'median_step_seconds':statistics.median(slowest),
                'tokens_per_second':tc.global_batch_tokens*args.steps/sum(slowest),
                'median_tokens_per_second':tc.global_batch_tokens/statistics.median(slowest),
                'losses':losses,'ranks':ranks,'validation_microbatch_check':result,
                'train_GFLOPs_per_token':theoretical_training_flops(mc)/1e9,
                'torch':torch.__version__,'cuda':torch.version.cuda,
                'data_manifest_sha256':manifest_hash,'source':source_manifest(),
                'stress_full_path':args.stress_full_path,'circle_stage_step':args.circle_stage_step,'last_step_stats':stats.tolist(),
                'scope':'real-data full-global-batch updates; excludes warmup, validation and checkpoint IO; fresh seed 42 per candidate, no formal training'}
        atomic_json(args.output,report)
        print(json.dumps({k:report[k] for k in ('micro_batch','accumulation','tokens_per_second','median_step_seconds')}),flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
