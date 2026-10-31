"""Pinned native parameter groups and schedules, with DDP gradient ownership."""
from collections import defaultdict
import math
import torch

from nexttransformer.training.sharded import ExperimentOptimizer

D12_SCALING_PARAMS = 110100912
D24_SCALING_PARAMS = 729810624


def build_optimizer(model, config):
    dscale = (model.config.width/768)**-.5
    bscale = math.sqrt(config.global_batch_tokens/524288)
    wd = config.weight_decay*bscale*D12_SCALING_PARAMS/D24_SCALING_PARAMS
    groups = []

    def adam(params, lr, betas, decay):
        params = list(params)
        if params: groups.append(dict(kind='adamw', params=params, lr=lr, initial_lr=lr,
                                      betas=betas, eps=1e-10, weight_decay=decay))

    adam(model.lm_head.parameters(), config.unembedding_lr*bscale*dscale, (.8,.96), .01)
    adam(model.transformer.wte.parameters(), config.embedding_lr*bscale*dscale, (.8,.995), .001)
    adam(model.value_embeds.parameters(), config.embedding_lr*bscale*dscale*.5, (.8,.995), .01)
    adam([model.resid_lambdas], config.scalar_lr*bscale*.01, (.8,.95), .05)
    adam([model.x0_lambdas], config.scalar_lr*bscale, (.96,.95), 0.)
    adam([model.smear_gate.weight, model.smear_lambda, model.backout_lambda], .2, (.8,.95), 0.)
    extras = [p for name, p in model.named_parameters() if name.startswith(('routers.', 'halter.'))]
    extra_scale = model.config.circle_halter_lr_multiplier if model.config.architecture == 'circle' else 1.
    adam(extras, config.adamw_lr*bscale*extra_scale, (.9,.95), 0.)
    if model.config.architecture == 'loopformer':
        adam(model.conditioner.parameters(),config.adamw_lr*bscale*model.config.loop_condition_lr_multiplier,
             (.9,.95),0.)
    allocated = {id(p) for group in groups for p in group['params']}
    matrices = defaultdict(list)
    scalars = []
    for p in model.parameters():
        if id(p) in allocated: continue
        if p.ndim == 2: matrices[tuple(p.shape)].append(p)
        else: scalars.append(p)
    adam(scalars, config.adamw_lr*bscale, (.9,.95), 0.)
    for shape in sorted(matrices):
        groups.append(dict(kind='muon', params=matrices[shape], lr=config.muon_lr*bscale,
                           initial_lr=config.muon_lr*bscale, momentum=.95, ns_steps=5, beta2=.9,
                           weight_decay=wd, initial_weight_decay=wd))
    ids = [id(p) for group in groups for p in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise RuntimeError('optimizer must cover all parameters exactly once')
    names = {id(p):name for name,p in model.named_parameters()}
    for group in groups: group['parameter_names'] = [names[id(p)] for p in group['params']]
    return ExperimentOptimizer(groups)


def clip_halter_gradients(model):
    """Clip only Circle's router group after DDP accumulation, before stepping."""
    if model.config.architecture != 'circle':return 0.
    return float(torch.nn.utils.clip_grad_norm_(model.halter.parameters(),
                 model.config.circle_halter_grad_clip,error_if_nonfinite=True))


def clip_condition_gradients(model):
    if model.config.architecture != 'loopformer':return 0.
    return float(torch.nn.utils.clip_grad_norm_(model.conditioner.parameters(),
                 model.config.loop_condition_grad_clip,error_if_nonfinite=True))


def set_schedule(optimizer, step, config):
    warmdown = round(config.warmdown_ratio*config.steps)
    start = config.steps-warmdown
    if step < config.warmup_steps: multiplier = (step+1)/max(1, config.warmup_steps)
    elif step <= start: multiplier = 1.
    else:
        progress = (config.steps-step)/max(1, warmdown)
        multiplier = progress+(1-progress)*config.final_lr_fraction
    if step < 400: momentum = .85+(.97-.85)*step/400
    elif step >= start: momentum = .97+(.90-.97)*(step-start)/max(1,warmdown)
    else: momentum = .97
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr']*multiplier
        if group['kind'] == 'muon':
            group['momentum'] = momentum
            group['weight_decay'] = group['initial_weight_decay']*.5*(1+math.cos(math.pi*step/config.steps))
    return multiplier
