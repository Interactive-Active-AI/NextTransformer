"""Shared train/benchmark accounting; Circle vectors are per block, not per loop."""
import torch


def training_stats(config, device):
    return torch.zeros(10+config.routing_steps, dtype=torch.float64, device=device)


def accumulate_stats(stats, out, path, accumulation, world, config):
    stats[0] += out['lm_loss'].detach()/accumulation/world
    stats[1] += out['router_loss'].detach()/accumulation/world
    stats[2] += 3*path['forward_flops']+path['recompute_flops']
    stats[3] += path['recompute_flops']
    stats[4] += path['input_tokens']
    stats[5] += path['active_depth_sum']
    for i, count in enumerate(path['active_tokens']): stats[6+i] += count
    start = 6+config.routing_steps
    stats[start] += out['loss'].detach()/accumulation/world
    stats[start+1] += path.get('logical_sparse_flops', path['forward_flops'])
    stats[start+2] += out.get('compute_loss', out['lm_loss'].new_zeros(())).detach()/accumulation/world
    stats[start+3] += out.get('objective', out['loss']).detach()/accumulation/world
