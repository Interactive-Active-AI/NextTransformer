"""Sparse Circle's gradients, causal attention, budgeting and resume contracts."""
import copy
from dataclasses import replace
import types
import pytest
import torch
from torch.nn import functional as F
from nexttransformer.models.backbone import norm,apply_rotary_emb
from nexttransformer.models.native.config import NativeModelConfig,NativeTrainConfig
from nexttransformer.models.native.depth import depth_reference
from nexttransformer.models.native.halter import DeltaDepthHalter,feature_forward_ops
from nexttransformer.models.native.model import NativeExperiment,build_model,theoretical_training_flops,window
from nexttransformer.native.optimizer import build_optimizer,clip_halter_gradients


def tiny(**kwargs):
    values=dict(architecture='circle',vocab_size=128,width=32,heads=2,mlp_width=128,
                sequence_len=16,pre_layers=1,core_layers=2,post_layers=1,circle_max_steps=6,
                circle_target_steps=4.,circle_random_warmup=0,circle_route_transition=0)
    return NativeModelConfig(**(values|kwargs))


def model(**kwargs):
    torch.manual_seed(42)
    m=build_model(tiny(**kwargs),torch.device('cpu'))
    for block in m.transformer.h:
        torch.nn.init.normal_(block.attn.c_proj.weight,std=.03)
        torch.nn.init.normal_(block.mlp.c_proj.weight,std=.03)
    return m


def dense_mask_reference(self,x,x0,idx,physical,virtual,conditioning=None,active=None):
    if active is None:
        return NativeExperiment._layer(self,x,x0,idx,physical,virtual,conditioning,None)
    c=self.config;b,n,d=x.shape;block=self.transformer.h[physical];attn=block.attn
    mixed=self.resid_lambdas[physical]*x+self.x0_lambdas[physical]*x0;z=norm(mixed)
    q=attn.c_q(z).view(b,n,c.heads,d//c.heads)
    k=attn.c_k(z).view_as(q);v=attn.c_v(z).view_as(q)
    if str(physical) in self.value_embeds:
        ve=self.value_embeds[str(physical)](idx).to(x.dtype).view_as(v)
        v=v+3*torch.sigmoid(attn.ve_gate(z[...,:12])).unsqueeze(-1)*ve
    cos,sin=self.cos[:,:n],self.sin[:,:n]
    q=norm(apply_rotary_emb(q,cos,sin))*1.2;k=norm(apply_rotary_emb(k,cos,sin))*1.2
    pos=torch.arange(n);distance=pos[:,None]-pos[None,:]
    allowed=(distance>=0)&(distance<=window(c,virtual)[0])
    allowed=allowed[None]&active[:,:,None]&active[:,None,:]
    y=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                                    attn_mask=allowed[:,None]).transpose(1,2).reshape(b,n,d)
    out=mixed+attn.c_proj(y);out=out+block.mlp(norm(out))
    return torch.where(active[...,None],out,x)


def test_reference_distribution_and_full_configuration():
    c=NativeModelConfig(architecture='circle');p,s=depth_reference(9,27,17.5)
    assert sum(p)==pytest.approx(1.) and sum(s)==pytest.approx(17.5)
    assert all(a>=b for a,b in zip(s,s[1:])) and all(x>0 for x in p)
    assert c.max_depth==33 and c.recursions==3 and (c.pre_layers,c.core_layers,c.post_layers)==(3,9,3)
    with torch.device('meta'):m=NativeExperiment(c)
    assert sum(p.numel() for p in m.parameters())==927992523
    assert sum(p.numel() for p in m.halter.parameters())==1555
    assert abs(theoretical_training_flops(c)/theoretical_training_flops(NativeModelConfig())-1)<.002
    assert feature_forward_ops(c)>0


@pytest.mark.parametrize('value,delta',[(0.,0.),(1.,0.),(1e-12,1e-12),(1.,.2)])
def test_finite_token_local_features_and_fp32(value,delta):
    h=DeltaDepthHalter(32,4);h.init_weights([.8,.7,.6,.5]);h.delta_scale.data.fill_(.2)
    before=torch.full((2,3,32),value,requires_grad=True)
    after=torch.full((2,3,32),value+delta,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):loss=h(before,after,2).sum()
    assert loss.dtype==torch.float32;loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters())
    assert torch.isfinite(before.grad).all() and torch.isfinite(after.grad).all()
    changed=after.detach().clone();changed[1,2]+=1
    torch.testing.assert_close(h(before.detach(),changed,2)[0],h(before.detach(),after.detach(),2)[0])


def test_sparse_matches_dense_masked_reference_outputs_routes_and_gradients():
    a=model();b=copy.deepcopy(a);b._layer=types.MethodType(dense_mask_reference,b)
    a.halter.depth_bias.data.copy_(torch.tensor([-.4,0.,.4,.8]));b.load_state_dict(a.state_dict())
    x=torch.randint(128,(3,16));y=x.roll(-1,1)
    left=a(x,y);right=b(x,y)
    assert a.last_stats['active_tokens']==b.last_stats['active_tokens']
    assert any(0<n<48 for n in a.last_stats['active_tokens'][2:])
    torch.testing.assert_close(left['loss'],right['loss'],rtol=2e-5,atol=2e-6)
    left['loss'].backward();right['loss'].backward()
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6,msg=name)


def test_sparse_empty_sequence_padding_and_original_window_positions():
    m=model(sequence_len=512);reference=copy.deepcopy(m)
    b,n,d=2,160,32;x=torch.randn(b,n,d,requires_grad=True);x0=torch.randn_like(x)
    idx=torch.randint(128,(b,n));active=torch.zeros((b,n),dtype=torch.bool)
    active[1,[0,2,130,159]]=True;m.last_stats=m._stats(idx)
    out=m._layer(x,x0,idx,2,4,active=active)
    expected=dense_mask_reference(reference,x,x0,idx,2,4,active=active)
    torch.testing.assert_close(out,expected,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(out[0],x[0],rtol=0,atol=0)
    out.square().mean().backward();assert torch.isfinite(x.grad).all()


def test_lm_gradient_reaches_selected_halter_without_budget_loss():
    m=model();m.halter.weight.data.zero_();m.halter.depth_bias.data.fill_(-.4)
    x=torch.randint(128,(2,16));out=m(x,x.roll(-1,1));out['lm_loss'].backward()
    assert m.last_stats['active_tokens']==[32]*6
    assert torch.isfinite(m.halter.depth_bias.grad).all()
    assert (m.halter.depth_bias.grad.abs()>1e-10).all()
    assert m.halter.delta_scale.grad.abs()>0


def test_rejected_tokens_skip_blocks_and_budget_reopens_first_closed_gate():
    m=model();m.halter.weight.data.zero_();m.halter.depth_bias.data.fill_(-2.)
    m.halter.depth_bias.data[0]=.4;calls=[]
    handle=m.transformer.h[1].attn.c_q.register_forward_hook(lambda *args:calls.append(1))
    x=torch.randint(128,(2,16));out=m(x,x.roll(-1,1));handle.remove()
    assert m.last_stats['active_tokens']==[32,32,0,0,0,0] and len(calls)==1
    out['router_loss'].backward()
    assert m.halter.depth_bias.grad[0]>0
    old=float(m.circle_budget.dual);m.update_route_budget(m.last_stats['active_tokens'],32,0)
    assert float(m.circle_budget.dual)<old
    with pytest.raises(ValueError,match='once'):m.update_route_budget([32,32,0,0,0,0],32,0)


def test_overspend_pushes_toward_exit_and_eval_preserves_buffers():
    m=model();m.halter.weight.data.zero_();m.halter.depth_bias.data.fill_(-.4)
    x=torch.randint(128,(2,16));m(x,x)['router_loss'].backward()
    assert m.halter.depth_bias.grad[0]<0
    m.update_route_budget(m.last_stats['active_tokens'],32,0);assert m.circle_budget.dual>0
    m.eval();saved=copy.deepcopy(m.state_dict());rng=m.routing_rng_state();m(x)
    for key,value in saved.items():torch.testing.assert_close(value,m.state_dict()[key],rtol=0,atol=0)
    assert m.routing_rng_state() is rng
    with pytest.raises(ValueError):m.update_route_budget([32]*6,32,1)


def test_curriculum_samples_reference_and_checkpointable_rng():
    m=model(circle_random_warmup=2,circle_route_transition=2)
    idx=torch.zeros((100,1000),dtype=torch.long);forced,depths=m._circle_random_routes(idx)
    assert forced.all() and depths.min()>=2 and depths.max()<=6
    assert float(depths.float().mean())==pytest.approx(4.,abs=.025)
    state=m.routing_rng_state();expected=m._circle_random_routes(idx)
    m.restore_routing_rng(state);actual=m._circle_random_routes(idx)
    assert all(torch.equal(x,y) for x,y in zip(actual,expected))
    m.set_circle_step(2);assert m.learned_route_fraction()==.5
    m.set_circle_step(3);assert m.circle_stage()=='learned'
    m.set_circle_step(0);m.eval();assert m.circle_stage()=='learned'


def test_optimizer_halter_rate_and_independent_clipping():
    m=model();tc=NativeTrainConfig();optimizer=build_optimizer(m,tc)
    groups=[g for g in optimizer.param_groups if any(n.startswith('halter.') for n in g['parameter_names'])]
    assert len(groups)==1 and groups[0]['kind']=='adamw'
    assert groups[0]['lr']==pytest.approx(tc.adamw_lr*2**.5*.5)
    for p in m.halter.parameters():p.grad=torch.ones_like(p)*10
    before=m.transformer.h[0].attn.c_q.weight;before.grad=torch.ones_like(before)
    assert clip_halter_gradients(m)>1
    assert torch.sqrt(sum(p.grad.square().sum() for p in m.halter.parameters()))<=1.00001
    assert torch.equal(before.grad,torch.ones_like(before))


def test_invalid_or_legacy_circle_configs_fail_explicitly():
    for changes in ({'circle_method':'st'},{'circle_halter':'delta_phase'},
                    {'circle_max_steps':4},{'circle_target_steps':2.},{'recursions':4},
                    {'circle_halter_eps':0.},{'circle_threshold':.7}):
        with pytest.raises(ValueError):replace(tiny(),**changes)
