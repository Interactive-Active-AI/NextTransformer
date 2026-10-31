"""Common-core execution and the concrete unbounded-conditioning failure mode."""
import types
import pytest
import torch
from nexttransformer.models.native.config import NativeModelConfig,NativeTrainConfig
from nexttransformer.models.native.model import build_model,NativeExperiment,theoretical_training_flops
from nexttransformer.native.optimizer import build_optimizer,clip_condition_gradients


def tiny():
    return NativeModelConfig(architecture='loopformer',vocab_size=128,width=32,heads=2,
                             mlp_width=128,sequence_len=16,pre_layers=1,core_layers=2,post_layers=1,fourier_dim=16)


def test_prefix_and_coda_run_once_core_twice():
    m=build_model(tiny(),torch.device('cpu'));calls=[]
    original=m._layer
    def observed(self,x,x0,idx,physical,virtual,conditioning=None,active=None):
        calls.append((physical,virtual,conditioning is not None))
        return original(x,x0,idx,physical,virtual,conditioning,active)
    m._layer=types.MethodType(observed,m);x=torch.randint(128,(2,16));m(x,x)
    assert calls==[(0,0,False),(1,1,True),(2,2,True),(1,3,True),(2,4,True),(3,5,False)]
    assert m.last_stats['active_depth_sum']==32*6
    assert m.last_stats['active_tokens']==[32,32]


def test_identity_initialization_preserves_first_projection_gradients():
    m=build_model(tiny(),torch.device('cpu'));x=torch.randint(128,(2,16))
    assert all(torch.equal(v,torch.zeros_like(v)) for v in m.conditioner(0,2,x.device))
    m(x,x.roll(-1,1))['loss'].backward()
    assert all(m.transformer.h[i].mlp.c_proj.weight.grad.abs().sum()>0 for i in (1,2))
    # After the backbone has nonzero residual outputs, conditioning also learns.
    for block in m.transformer.h:torch.nn.init.normal_(block.mlp.c_proj.weight,std=.03)
    m.zero_grad();m(x,x.roll(-1,1))['loss'].backward()
    assert all(v[-1].weight.grad.abs().sum()>0 for v in m.conditioner.modulation)


def test_large_condition_weights_cannot_create_unbounded_or_negative_scales():
    m=build_model(tiny(),torch.device('cpu'))
    with torch.no_grad():
        for p in m.conditioner.parameters():p.fill_(1000.)
    with torch.autocast('cpu',dtype=torch.bfloat16):values=m.conditioner(1,2,torch.device('cpu'))
    assert all(v.dtype==torch.float32 for v in values)
    for v in values:
        assert torch.isfinite(v).all()
        assert (1+v).min()>=.5 and (1+v).max()<=1.5
    sum(x.sum() for x in values).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.conditioner.parameters())


def test_conditioning_uses_adamw_not_muon_and_clips_independently():
    m=build_model(tiny(),torch.device('cpu'));tc=NativeTrainConfig();opt=build_optimizer(m,tc)
    group=next(g for g in opt.param_groups if any(n.startswith('conditioner.') for n in g['parameter_names']))
    assert group['kind']=='adamw' and group['lr']==pytest.approx(.0006*2**.5*.5)
    assert all(n.startswith('conditioner.') for n in group['parameter_names'])
    for p in m.conditioner.parameters():p.grad=torch.ones_like(p)*10
    block=m.transformer.h[1].attn.c_q.weight;block.grad=torch.ones_like(block)
    assert clip_condition_gradients(m)>1
    assert sum(p.grad.square().sum() for p in m.conditioner.parameters()).sqrt()<=1.00001
    assert torch.equal(block.grad,torch.ones_like(block))


def test_shared_physical_architecture_and_matched_compute():
    dense=theoretical_training_flops(NativeModelConfig())
    for name in ('mor','loopformer','circle'):
        c=NativeModelConfig(architecture=name)
        assert (c.pre_layers,c.core_layers,c.post_layers)==(3,9,3) and c.n_layer==15
        assert abs(theoretical_training_flops(c)/dense-1)<.02
        with torch.device('meta'):m=NativeExperiment(c)
        assert len(m.transformer.h)==15
