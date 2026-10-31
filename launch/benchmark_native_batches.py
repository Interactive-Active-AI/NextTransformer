"""Single-host batch sweep controller with sampled NVIDIA telemetry and bounded OOM handling.

Run after sourcing launch/env.sh. All trials share seed and global batch.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time


def atomic_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def telemetry_summary(path,result):
    ranks=result['ranks']
    # Restrict to the common steady training interval, excluding warmup/validation.
    start=max(r['measurement_start'] for r in ranks);end=min(r['measurement_end'] for r in ranks)
    samples=[json.loads(line) for line in path.read_text().splitlines()]
    summaries=[]
    for rank in ranks:
        matching=[s for s in samples if s['uuid'].removeprefix('GPU-').lower()==rank['uuid'].removeprefix('GPU-').lower()]
        if not matching:raise ValueError(f'no telemetry for rank {rank["rank"]} UUID {rank["uuid"]}')
        steady=[s for s in matching if start<=s['time']<=end]
        if len(steady)<5:raise ValueError('not enough steady telemetry samples')
        summaries.append({'rank':rank['rank'],'gpu_index':steady[0]['index'],'samples':len(steady),
                          'gpu_util_mean_pct':statistics.mean(s['gpu_util'] for s in steady),
                          'gpu_util_min_pct':min(s['gpu_util'] for s in steady),
                          'gpu_util_max_pct':max(s['gpu_util'] for s in steady),
                          'memory_used_peak_mib':max(s['memory_used'] for s in steady),
                          'memory_used_peak_pct':100*max(s['memory_used']/s['memory_total'] for s in steady),
                          'memory_used_peak_mib_entire_trial':max(s['memory_used'] for s in matching),
                          'power_mean_w':statistics.mean(s['power'] for s in steady),
                          'sm_clock_mean_mhz':statistics.mean(s['sm_clock'] for s in steady)})
    return {'sample_period_ms':200,'interval_start':start,'interval_end':end,'per_gpu':summaries,
            'note':'nvidia-smi GPU busy percentage; not SM occupancy or FLOP efficiency. Memory used includes allocator reserves and CUDA/NCCL overhead; peaks are sampled.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['dense','mor','loopformer','circle'],choices=['dense','mor','loopformer','circle'])
    p.add_argument('--batches',nargs='+',type=int,default=[1,2,4,8,16])
    p.add_argument('--gpus',type=int,default=8)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--confirm-steps',type=int,default=40)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--stress-full-path',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if min(args.gpus,args.warmup,args.steps,args.confirm_steps,args.timeout)<=0 or min(args.warmup,args.steps,args.confirm_steps)<2:
        p.error('GPU count and timeout must be positive; warmup/measurement lengths must be >= 2')
    if len(set(args.models))!=len(args.models) or len(set(args.batches))!=len(args.batches):
        p.error('models and batches must not contain duplicates')
    for model in args.models:
        config=json.loads((root/'configs/native'/f'{model}.json').read_text())
        for batch in args.batches:
            denominator=args.gpus*batch*config['model']['sequence_len']
            if denominator<=0 or config['train']['global_batch_tokens']%denominator:
                p.error(f'{model} batch {batch} does not divide the fixed global token batch')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=args.output/'status.json'
    if status_path.exists():raise ValueError('use a fresh sweep directory; never overwrite a previous sweep')
    inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if inventory:raise RuntimeError('GPUs have active compute processes; wait for the current jobs before tuning')
    env={**os.environ,'WANDB_MODE':'disabled','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'}
    trials=[]
    def run(model,batch,phase,steps):
        name=f'{model}_b{batch}_{phase}';directory=args.output/name;directory.mkdir()
        result_path=directory/'result.json';sample_path=directory/'telemetry.jsonl'
        atomic_json(status_path,{'phase':'running','trial':name,'completed':trials})
        query='index,uuid,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm'
        monitor=subprocess.Popen(['nvidia-smi',f'--query-gpu={query}','--format=csv,noheader,nounits','-lms','200'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
        sampling_errors=[]
        def sample():
            with sample_path.open('w') as f:
                for line in monitor.stdout:
                    try:
                        values=next(csv.reader([line]));idx,uuid=values[:2];nums=[float(x.strip()) for x in values[2:]]
                        row=dict(zip(('memory_total','memory_used','gpu_util','memory_util','power','temperature','sm_clock'),nums))
                        f.write(json.dumps({'time':time.time(),'index':int(idx),'uuid':uuid.strip(),**row})+'\n')
                    except (ValueError,IndexError) as exc:sampling_errors.append(str(exc))
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={args.gpus}',
                 '-m','nexttransformer.native.benchmark','--config',str(root/'configs/native'/f'{model}.json'),
                 '--micro-batch',str(batch),'--warmup',str(args.warmup),'--steps',str(steps),'--output',str(result_path)]
        if args.stress_full_path: command.append('--stress-full-path')
        started=time.time()
        try:
            with (directory/'process.log').open('w') as log:
                worker=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=worker.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid,signal.SIGTERM)
                    try:worker.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
                    raise RuntimeError(f'trial timeout: {name}')
        finally:
            monitor.terminate();monitor.wait(timeout=10);thread.join(timeout=10)
        entry={'model':model,'micro_batch':batch,'phase':phase,'exit_code':code,'seconds':time.time()-started}
        if code:
            log=(directory/'process.log').read_text()
            if 'OutOfMemoryError' in log or 'CUDA out of memory' in log:
                entry['status']='oom'
                entry['reason']=next((line.strip() for line in log.splitlines() if 'out of memory' in line.lower()),'CUDA OOM')
            else:raise RuntimeError(f'{name} failed: '+log[-4000:])
        else:
            if sampling_errors:raise RuntimeError(f'telemetry parsing errors: {sampling_errors[:2]}')
            result=json.loads(result_path.read_text());entry['status']='passed'
            entry['result']=result;entry['telemetry']=telemetry_summary(sample_path,result)
        atomic_json(directory/'summary.json',entry);trials.append(entry)
        atomic_json(status_path,{'phase':'running','trial':name,'completed':trials})
        print(json.dumps({'trial':name,'status':entry['status'],'tokens_per_second':entry.get('result',{}).get('tokens_per_second')}),flush=True)
        return entry
    try:
        selections={}
        for model in args.models:
            for batch in args.batches:run(model,batch,'sweep',args.steps)
            good=[t for t in trials if t['model']==model and t['status']=='passed']
            if not good:raise RuntimeError(f'no fitting batch for {model}')
            best=max(t['result']['tokens_per_second'] for t in good)
            # Prefer fewer accumulations when throughput is within 2% of the best.
            choice=max((t for t in good if t['result']['tokens_per_second']>=.98*best),key=lambda t:t['micro_batch'])
            confirmed=run(model,choice['micro_batch'],'confirm',args.confirm_steps)
            if confirmed['status']!='passed':raise RuntimeError('selected batch failed confirmation')
            selections[model]={'selected_batch':choice['micro_batch'],'largest_fitting_batch':max(t['micro_batch'] for t in good),
                               'accumulation':confirmed['result']['accumulation']}
        atomic_json(args.output/'result.json',{'phase':'complete','selection_rule':'largest batch within 2% of the best sweep throughput, followed by longer confirmation','selections':selections,'trials':trials})
        atomic_json(status_path,{'phase':'complete','selections':selections})
        print('BATCH_SWEEP_COMPLETE '+json.dumps(selections),flush=True)
    except BaseException as exc:
        atomic_json(status_path,{'phase':'failed','error':repr(exc),'completed':trials});raise

if __name__=='__main__':main()
