from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from nexttransformer.models.native.config import NativeTrainConfig
from nexttransformer.native.optimizer import build_optimizer as native_optimizer
from nexttransformer.runtime.checkpoint import save_checkpoint,load_checkpoint
from nexttransformer.protocol.spec import Protocol
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups,build_optimizer,transition_stage
from nexttransformer.protocol.admission import BudgetState, events_at
from nexttransformer.evaluation.formal import EvalRows


def test_schedule_budget_resume_and_final_admission():
    p=Protocol();state=BudgetState();state.record_eval("val")
    assert sum(events_at(s)["val"] for s in range(p.steps+1))==189
    assert sum(events_at(s)["core"] for s in range(p.steps+1))==6
    with pytest.raises(ValueError):state.advance(p.global_batch_tokens//2,100)
    for step in range(1,p.steps+1):
        state.advance(p.global_batch_tokens,p.global_batch_tokens*7)
        for kind in ("val","core"):
            if events_at(step)[kind]:state.record_eval(kind)
    restored=BudgetState.from_state_dict(state.state_dict())
    assert restored.final_status(7)=="matched"
    assert restored.final_status(8)=="budget_mismatch"
    restored.core_steps.pop();assert restored.final_status(7)=="incomplete_eval"


def test_native_optimizer_group_parity_and_hyperparameter_guard():
    m=build_model(debug_config("dense"));actual=parameter_groups(m)
    config=m.config;m.config=SimpleNamespace(width=config.width,architecture="dense")
    expected=native_optimizer(m,NativeTrainConfig()).param_groups;m.config=config
    def signature(group):
        return {key:group[key] for key in ("kind","parameter_names","lr","initial_lr","weight_decay",
                *( ("betas","eps") if group["kind"]=="adamw" else ("momentum","ns_steps","beta2") ))}
    assert [signature(g) for g in actual]==[signature(g) for g in expected]
    with pytest.raises(ValueError,match="controlled training field"):
        build_optimizer(m,replace(NativeTrainConfig(),muon_lr=.03))


def test_huginn_rows_are_invariant_to_microbatch_and_restorable():
    c=debug_config("huginn");m=build_model(c);m.eval()
    with torch.no_grad():
        for block in m.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03);block.mlp.c_proj.weight.normal_(std=.03)
        x=torch.randint(c.vocab_size,(4,c.sequence_len));m.set_row_offset(31)
        state=m.routing_rng_state();all_rows=m(x)
        m.set_row_offset(31);first=m(x[:2]);m.set_row_offset(33);second=m(x[2:])
        torch.testing.assert_close(torch.cat((first,second)),all_rows,atol=2e-6,rtol=2e-6)
        m.restore_routing_rng(state);torch.testing.assert_close(m(x),all_rows,rtol=0,atol=0)


def test_ouro_transition_preserves_gate_moments_and_schedule():
    c=debug_config("ouro");m=build_model(c);opt=build_optimizer(m)
    old={"step":123,"exp_avg":torch.ones_like(m.halter.weight),"exp_avg_sq":torch.ones_like(m.halter.weight)*2}
    opt.state[m.halter.weight]=old
    step=Protocol().steps-c.ouro_gate_steps
    new,changed=transition_stage(m,opt,step)
    assert changed and new.state[m.halter.weight] is old
    assert {n for g in new.param_groups for n in g["parameter_names"]}=={"halter.weight","halter.bias"}
    assert new.param_groups[0]["lr"]<new.param_groups[0]["initial_lr"]


@pytest.mark.parametrize("method,fields",[("kda",(".log_decay",".dt_bias")),("nsa",(".compress_position",)),
                                          ("gdn",(".log_decay",".dt_bias",".decay_proj.",".beta_proj."))])
def test_gate_and_position_tables_use_adam_not_muon(method,fields):
    m=build_model(debug_config(method))
    ownership={name:group["kind"] for group in parameter_groups(m) for name in group["parameter_names"]}
    for field in fields:
        selected=[kind for name,kind in ownership.items() if field in name]
        assert selected and set(selected)=={"adamw"}


def test_reference_can_use_existing_atomic_checkpoint(tmp_path):
    class Stream:
        def __init__(self):self.rows=64
        def state_dict(self):return {"rows":self.rows}
        def load_state_dict(self,state):self.rows=state["rows"]
    c=debug_config("huginn");m=build_model(c);m.set_step(7);m.set_row_offset(3590)
    opt=build_optimizer(m);stream=Stream()
    saved={n:p.detach().clone() for n,p in m.named_parameters()}
    path=save_checkpoint(tmp_path,m,opt,stream,c,NativeTrainConfig(),{"step":7},0,1)
    with torch.no_grad():m.lm_head.weight.add_(1)
    m.set_step(99);stream.rows=999
    load_checkpoint(path,m,opt,stream,0,1)
    assert stream.rows==64 and m.optimizer_step==7 and m.row_offset==3590
    for n,p in m.named_parameters():torch.testing.assert_close(p,saved[n],rtol=0,atol=0)
