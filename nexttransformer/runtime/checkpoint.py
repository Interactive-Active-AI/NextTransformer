"""Atomic native checkpoints include exact per-rank best-fit buffers and cursors."""
from pathlib import Path
from dataclasses import asdict
import json
import torch
import torch.distributed as dist
from nexttransformer.runtime.files import atomic_json, sha256
from nexttransformer.runtime.process import rng_state, restore_rng


def save_checkpoint(output, model, optimizer, stream, mc, tc, state, rank, world):
    directory = Path(output)/f'checkpoint_{state["step"]:08d}'
    # Never leave an old completion marker visible while replacing its payload.
    # Rank zero creates the directory before the other ranks write their shard.
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=False)
    if world > 1: dist.barrier()
    temporary = directory/f'rank_{rank:04d}.pt.tmp'
    torch.save({'optimizer':optimizer.state_dict(), 'rng':rng_state(), 'data':stream.state_dict(),
                'routing_rng':model.routing_rng_state()}, temporary)
    temporary.replace(directory/f'rank_{rank:04d}.pt')
    if rank == 0:
        temporary = directory/'model.pt.tmp'
        torch.save({'model':model.state_dict(), 'model_config':asdict(mc), 'train_config':asdict(tc),
                    'state':state, 'world_size':world}, temporary)
        temporary.replace(directory/'model.pt')
    if world > 1: dist.barrier()
    if rank == 0:
        atomic_json(directory/'complete.json', {'step':state['step'], 'world_size':world,
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(directory.glob('*.pt'))}})
        atomic_json(Path(output)/'latest.json', {'step':state['step'], 'checkpoint':directory.name})
    if world > 1: dist.barrier()
    return directory


def load_checkpoint(path, model, optimizer=None, stream=None, rank=0, world=1):
    path = Path(path)
    manifest = json.loads((path/'complete.json').read_text())
    if optimizer is not None and manifest['world_size'] != world:
        raise ValueError('optimizer and data sharding require original world size')
    needed = ['model.pt']+([f'rank_{rank:04d}.pt'] if optimizer is not None else [])
    for name in needed:
        item = manifest['files'][name]
        if (path/name).stat().st_size != item['bytes'] or sha256(path/name) != item['sha256']:
            raise ValueError(f'checkpoint checksum mismatch: {name}')
    saved = torch.load(path/'model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    if optimizer is not None:
        local = torch.load(path/f'rank_{rank:04d}.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(local['optimizer'])
        stream.load_state_dict(local['data'])
        restore_rng(local['rng'])
        model.restore_routing_rng(local['routing_rng'])
    return saved
