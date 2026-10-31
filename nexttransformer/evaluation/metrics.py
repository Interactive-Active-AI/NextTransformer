"""Route accounting and pinned CORE bundle integrity."""
import json
from pathlib import Path
import torch
from nexttransformer.runtime.files import sha256, CORE_SHA256
from nexttransformer.runtime.process import reduce_sum

class Metrics:
    def __init__(self, steps=4, circle=False):
        self.steps=steps
        self.circle=circle
        self.values=[0.0]*(3+steps)

    def add(self, stats):
        self.values[0]+=stats['input_tokens']; self.values[1]+=stats['forward_flops']
        self.values[2]+=stats['active_depth_sum']
        for i,n in enumerate(stats['active_tokens']): self.values[3+i]+=n

    def result(self, device):
        values=reduce_sum(torch.tensor(self.values,dtype=torch.float64,device=device)).tolist()
        n=max(1,values[0])
        result = {'input_tokens':int(values[0]),'forward_flops_estimate':values[1],
                'mean_active_depth':values[2]/n,'active_ratios':[v/n for v in values[3:]],
                ('mean_ring_steps' if self.circle else 'mean_recursions'):sum(values[3:])/n}
        if self.circle:
            counts=values[3:]+[0]
            result['exit_ratios']=[(counts[i]-counts[i+1])/n for i in range(self.steps)]
            result['cap_ratio']=counts[-2]/n
        return result

def verify_core(resources):
    resources=Path(resources)
    pin=json.loads((resources/'core_manifest.json').read_text())
    if pin['archive_sha256']!=CORE_SHA256: raise ValueError('CORE archive differs from the experiment protocol')
    if pin['entries']!=22: raise ValueError('CORE must contain 22 entries')
    for name,digest in pin['files'].items():
        if sha256(resources/'eval_bundle'/name)!=digest:
            raise ValueError(f'CORE resource checksum mismatch: {name}')
    return pin
