"""Bounded full-path 8-GPU preflight; halve microbatch on OOM, keep global tokens."""
from nexttransformer.runtime.site import bootstrap_runtime, add_runtime_arguments
if __name__ == '__main__': bootstrap_runtime()
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from nexttransformer.runtime.files import atomic_json
from nexttransformer.models.native.config import load_config


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    add_runtime_arguments(p)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=1800)
    args=p.parse_args()
    mc,tc=load_config(args.config)
    if args.gpus!=8:p.error('native hardware preflight requires eight GPUs')
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if running:raise RuntimeError('GPU compute processes already exist; preflight will not interrupt them')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise ValueError('preflight output already exists')
    batch=args.micro_batch or 8
    attempts=[]
    while batch>=1:
        tc.accumulation(mc,args.gpus,batch)
        result=args.output.with_name(args.output.stem+f'_b{batch}.json')
        log=result.with_suffix('.log')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                 '-m','nexttransformer.native.benchmark','--config',str(args.config),'--micro-batch',str(batch),
                 '--warmup','2','--steps','2','--stress-full-path','--output',str(result)]
        with log.open('w') as out:
            worker=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                                    env={**os.environ,'WANDB_MODE':'disabled'})
            try:code=worker.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGTERM)
                try:worker.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
                raise RuntimeError(f'preflight timed out; inspect {log}')
        attempts.append({'micro_batch':batch,'exit_code':code,'log':str(log)})
        if code==0:
            atomic_json(args.output,{'status':'passed','micro_batch':batch,
                'accumulation':tc.accumulation(mc,args.gpus,batch),'global_batch_tokens':tc.global_batch_tokens,
                'attempts':attempts,'evidence':json.loads(result.read_text())})
            print(f'PREFLIGHT PASSED: --micro-batch {batch}; accumulation {tc.accumulation(mc,8,batch)}')
            return
        text=log.read_text()
        if 'out of memory' not in text.lower() and 'OutOfMemoryError' not in text:
            raise RuntimeError(f'preflight failed for a reason other than OOM; inspect {log}')
        batch//=2
    atomic_json(args.output,{'status':'failed','attempts':attempts})
    raise RuntimeError('even microbatch=1 did not fit')


if __name__=='__main__':main()
