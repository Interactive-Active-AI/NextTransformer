"""Pinned native Muon/AdamW groups with explicit method-auxiliary ownership."""
from collections import defaultdict
import math

from nexttransformer.training.sharded import ExperimentOptimizer
from nexttransformer.models.native.config import NativeTrainConfig
from nexttransformer.native.optimizer import D12_SCALING_PARAMS, D24_SCALING_PARAMS, set_schedule
from nexttransformer.protocol.spec import Protocol


def parameter_groups(model, train=None):
    c=train or NativeTrainConfig()
    Protocol().validate_train_config(c)
    dscale=(model.config.width/768)**-.5
    bscale=math.sqrt(c.global_batch_tokens/524288)
    groups=[]

    def adam(params,lr,betas,decay):
        params=[p for p in params if p.requires_grad]
        if params:groups.append(dict(kind="adamw",params=params,lr=lr,initial_lr=lr,
                                    betas=betas,eps=1e-10,weight_decay=decay))

    adam(model.lm_head.parameters(),c.unembedding_lr*bscale*dscale,(.8,.96),.01)
    adam(model.transformer.wte.parameters(),c.embedding_lr*bscale*dscale,(.8,.995),.001)
    adam(model.value_embeds.parameters(),c.embedding_lr*bscale*dscale*.5,(.8,.995),.01)
    adam([model.resid_lambdas],c.scalar_lr*bscale*.01,(.8,.95),.05)
    adam([model.x0_lambdas],c.scalar_lr*bscale,(.96,.95),0.)
    adam([model.smear_gate.weight,model.smear_lambda,model.backout_lambda],.2,(.8,.95),0.)
    auxiliary=("halter.",".index_",".beta_proj.",".gate_down.",".gate_up.",".decay_proj.",
               ".slot_proj.",".branch_gate.",".output_gate.",
               ".log_decay",".dt_bias",".compress_position","residual_maps.","depth_queries")
    extras=[p for name,p in model.named_parameters() if any(key in name for key in auxiliary)]
    adam(extras,c.adamw_lr*bscale,(.9,.95),0.)
    allocated={id(p) for group in groups for p in group["params"]}
    matrices=defaultdict(list);other=[]
    for p in model.parameters():
        if not p.requires_grad or id(p) in allocated:continue
        if p.ndim==2:matrices[tuple(p.shape)].append(p)
        else:other.append(p)
    adam(other,c.adamw_lr*bscale,(.9,.95),0.)
    wd=c.weight_decay*bscale*D12_SCALING_PARAMS/D24_SCALING_PARAMS
    for shape in sorted(matrices):
        groups.append(dict(kind="muon",params=matrices[shape],lr=c.muon_lr*bscale,
                           initial_lr=c.muon_lr*bscale,momentum=.95,ns_steps=5,beta2=.9,
                           weight_decay=wd,initial_weight_decay=wd))
    names={id(p):name for name,p in model.named_parameters()}
    ids=[id(p) for group in groups for p in group["params"]]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("trainable parameters must have exactly one optimizer owner")
    for group in groups:group["parameter_names"]=[names[id(p)] for p in group["params"]]
    return groups


def build_optimizer(model, train=None):
    return ExperimentOptimizer(parameter_groups(model,train))


def transition_stage(model,optimizer,step,train=None):
    """Rebuild Ouro's trainable groups while retaining the gate's Adam state.

    Call at an optimizer boundary; recreate the DDP wrapper when changed=True.
    The caller retains global BudgetState, stream, scheduler step and RNG state.
    """
    previous={id(p) for group in optimizer.param_groups for p in group["params"]}
    model.set_training_stage(step)
    current={id(p) for p in model.parameters() if p.requires_grad}
    if previous==current:
        set_schedule(optimizer,step,train or NativeTrainConfig());return optimizer,False
    updated=build_optimizer(model,train)
    for group in updated.param_groups:
        for param in group["params"]:
            if param in optimizer.state:updated.state[param]=optimizer.state[param]
    set_schedule(updated,step,train or NativeTrainConfig())
    return updated,True
