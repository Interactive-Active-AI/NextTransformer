"""Independent arithmetic and configuration admission checks."""
from dataclasses import replace
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from nexttransformer.models.config import METHODS, debug_config
from nexttransformer.protocol.compute import Ledger,audit_graph,causal_edges,selected_edges
from nexttransformer.models.sparse import block_indices,qsa_indices,pool_teacher
from nexttransformer.models.linear import recurrent_attention


def test_matrix_forward_backward_arithmetic():
    ledger=Ledger()
    with FakeTensorMode(),ledger:
        a=torch.empty(7,5,requires_grad=True);b=torch.empty(5,3,requires_grad=True)
        y=a@b
        ledger.phase="backward";y.sum().backward()
    assert ledger.report()["matrix_total"]==3*2*7*5*3
    assert not ledger.unknown


def test_causal_window_off_by_one():
    for length in (4,17,2048):
        for left in (-1,0,4,512):
            explicit=sum(1 for i in range(length) for j in range(i+1) if left<0 or i-j<=left)
            assert causal_edges(length,left)==explicit


@pytest.mark.parametrize("method",["qsa","seer","moba","nsa"])
def test_selected_count_matches_real_mask(method):
    c=debug_config(method);t=17
    size=c.compression_size if method in ("qsa","nsa") else c.block_size
    count=(t+size-1)//size
    top=max(0,c.sparse_topk//size-1) if method=="moba" else max(1,(c.sparse_topk+size-1)//size)
    positions=torch.arange(t)
    scores=torch.randn(1,1,t,count)
    complete=(torch.arange(count)+1)*size-1<=positions[:,None]
    if method=="qsa":indices,valid=qsa_indices(scores,complete,positions,size,t,c.sparse_topk)
    else:indices,valid=block_indices(scores,complete,positions,size,t,top)
    assert int(valid.sum())==selected_edges(c,0,t,"joint")


def test_qsa_complete_boundary_and_teacher_max_pool():
    positions=torch.tensor([3,4,7]);size=4;t=8
    scores=torch.tensor([[[[9.,1.],[9.,1.],[9.,1.]]]])
    complete=(torch.arange(2)+1)*size-1<=positions[:,None]
    indices,valid=qsa_indices(scores,complete,positions,size,t,4)
    assert [indices[0,0,i][valid[0,0,i]].tolist() for i in range(3)]==[[0,1,2,3],[0,1,2,3,4],[0,1,2,3]]
    teacher=torch.tensor([[[.05,.3,.1,.05,.1,.1,.2,.1]]])
    torch.testing.assert_close(pool_teacher(teacher,4,2,"qsa"),torch.tensor([[[.3,.2]]]))


@pytest.mark.parametrize("method",METHODS)
def test_no_uncovered_float_ops(method):
    report=audit_graph(debug_config(method),depth=3 if method=="huginn" else None)
    assert not report["unknown"]
    assert report["phases"]["backward"]>0 and report["phases"]["optimizer"]>0
    assert report["padded_envelope"]>=report["total"]


def test_gate_stage_reduces_backward_not_forward():
    c=debug_config("ouro")
    joint=audit_graph(c);gate=audit_graph(c,stage="gate_only")
    assert not gate["unknown"]
    assert gate["phases"]["forward"]>joint["phases"]["forward"]*.9
    assert gate["phases"]["backward"]<joint["phases"]["backward"]*.1
    assert gate["phases"]["optimizer"]<joint["phases"]["optimizer"]*.01


def test_kda_all_gate_and_initial_state_gradients():
    b,t,h,d=1,3,1,2
    q,k,v=(torch.randn(b,t,h,d,dtype=torch.float64,requires_grad=True) for _ in range(3))
    g=torch.full_like(q,.8,requires_grad=True)
    beta=torch.full((b,t,h),.4,dtype=q.dtype,requires_grad=True)
    initial=torch.randn(b,h,d,d,dtype=q.dtype,requires_grad=True)
    fn=lambda q,k,v,g,beta,initial:recurrent_attention(q,k,v,kind="kda",decay=g,beta=beta,initial_state=initial)
    assert torch.autograd.gradcheck(fn,(q,k,v,g,beta,initial))


def test_gdn_head_gate_and_initial_state_gradients():
    """GDN reuses the KDA first-order BPTT with its head gate broadcast over key features."""
    b,t,h,d=1,3,2,2
    q,k,v=(torch.randn(b,t,h,d,dtype=torch.float64,requires_grad=True) for _ in range(3))
    g=(torch.rand(b,t,h,1,dtype=torch.float64)*.5+.4).requires_grad_()
    beta=torch.full((b,t,h),.4,dtype=q.dtype,requires_grad=True)
    initial=torch.randn(b,h,d,d,dtype=q.dtype,requires_grad=True)
    fn=lambda q,k,v,g,beta,initial:recurrent_attention(q,k,v,kind="gdn",decay=g,beta=beta,initial_state=initial)
    assert torch.autograd.gradcheck(fn,(q,k,v,g,beta,initial))


def test_chunk_tile_operators_charge_causal_pairs():
    c=4;pairs=c*(c+1)//2;n,kdim=3,5
    ledger=Ledger(pair_block=c)
    with FakeTensorMode(),ledger:
        q=torch.empty(n,c,kdim,requires_grad=True);k=torch.empty(n,c,kdim,requires_grad=True)
        scores=q@k.transpose(-1,-2)
        row=scores.sum(-1)
    ops=ledger.report()["operators"]
    mm=next(v for name,v in ops.items() if "bmm" in name)
    assert mm["theory"]==2*n*pairs*kdim and mm["padded_envelope"]==2*n*c*c*kdim
    total=next(v for name,v in ops.items() if "sum" in name)
    assert total["theory"]==n*(pairs-c)
    plain=Ledger()  # no chunk backend: the same graph is charged densely, as in v1
    with FakeTensorMode(),plain:
        (torch.empty(n,c,kdim)@torch.empty(n,kdim,c)).sum(-1)
    assert plain.report()["total"]==2*n*c*c*kdim+n*(c*c-c)


def test_cumsum_and_triangular_solve_rules():
    ledger=Ledger()
    with FakeTensorMode(),ledger:
        torch.empty(2,7,3).cumsum(1)
        torch.linalg.solve_triangular(torch.empty(4,6,6),torch.empty(4,6,5),upper=False,unitriangular=True)
        torch.linalg.solve_triangular(torch.empty(6,6),torch.empty(6,5),upper=False)
    ops=ledger.report()["operators"]
    assert sum(v["theory"] for name,v in ops.items() if "cumsum" in name)==2*7*3-2*3
    solves=sorted(v["theory"] for name,v in ops.items() if "solve_triangular" in name)
    # Same operator name, so both calls share one entry: unit 4x5 columns, non-unit adds n divisions.
    assert solves==[4*5*6*(6-1)+5*6*(6-1)+5*6]


@pytest.mark.parametrize("kind",["elu_linear","elu_norm","elu_qknorm"])
def test_elu_chunk_forward_matches_closed_form(kind):
    """Hand count of the registered chunk numerator plus the FP32 normalizer (elu_linear only)."""
    from nexttransformer.models.linear import chunk_attention
    b,t,h,d,c=2,24,3,5,8;chunks=t//c;pairs=c*(c+1)//2
    ledger=Ledger(pair_block=c)
    with FakeTensorMode(),ledger:
        q,k,v=(torch.empty(b,t,h,d) for _ in range(3))
        chunk_attention(q,k,v,kind=kind,chunk=c)
    tiles=b*h*chunks
    # Causal tiles, then reads for chunks 1..N-1, updates for 0..N-2 (adds from the second).
    numerator=tiles*4*pairs*d+b*h*((chunks-1)*(4*c*d*d+c*d)+(chunks-2)*d*d)
    normalizer=(b*t*h*d-b*h*d)+b*t*h*d+b*t*h*(d-1)+b*t*h+b*t*h*d
    assert ledger.report()["phases"]["forward"]==numerator+(normalizer if kind=="elu_linear" else 0)


@pytest.mark.parametrize("method",["deltanet","kda","retnet","elu_linear","elu_norm","elu_qknorm","gdn"])
def test_chunk_backend_audit_is_covered_and_affine_in_chunks(method):
    # Debug vocabulary 64 would equal the chunk length (ambiguous tiles): use 96.
    c=replace(debug_config(method,sequence_len=320),backend="fla_chunk64",vocab_size=96)
    with pytest.raises(ValueError,match="chunk length"):audit_graph(replace(c,vocab_size=64),sequence_len=128)
    reports=[audit_graph(c,sequence_len=t,batch=2) for t in (192,256,320)]
    for r in reports:
        assert not r["unknown"] and r["pair_block"]==64
        assert r["padded_envelope"]>r["total"]
    a,b,test=(r["total"] for r in reports)
    assert test==2*b-a
