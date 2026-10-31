"""Versioned, shape-only forward/backward/optimizer arithmetic accounting.

This is a mathematical cost model, not a GPU profiler. FakeTensor executes
autograd and the actual first-order reference backward without allocating model
storage. Attention has explicit causal/selected-edge rules; its padded execution
envelope is reported separately. Chunked linear backends are audited through
their plain-torch chunk algorithm: operators on a C x C intra-chunk tile are
charged for causal pairs only (i >= j, the `causal_edges` rule), the dense tile
goes to the padded envelope, and kernel-internal recomputation is outside the
theory account. Unknown floating operators fail admission.

v2 adds cumsum, unit/non-unit triangular solves, zero-cost tril/triu/eye and the
chunk pair rule; graphs without these operators count exactly as in v1.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace
import math

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode

from nanochat import flash_attention
from nexttransformer.models.backbone import CausalSelfAttention
from nanochat.optim import adamw_step_fused, muon_step_fused
from nexttransformer.models.config import CHUNK_BACKENDS
from nexttransformer.protocol.spec import Protocol
from nexttransformer.models.lm import build_model
from nexttransformer.models import sparse
from nexttransformer.models.plugins import stage_start
from nexttransformer.models.registry import PLUGIN_METHODS
from nexttransformer.training.optimizer import parameter_groups

FORMULA_VERSION="reference-aten-math-v2"


def causal_edges(length, left):
    """left is a count of previous keys: width includes the current key."""
    width=length if left<0 else min(length,left+1)
    return width*(width+1)//2+(length-width)*width


def selected_edges(config, start, stop, stage, branch=0, kind=None):
    kind=kind or config.method
    if stage=="burnin" and kind in ("dsa","qsa","seer"):
        return sum(i+1 for i in range(start,stop))
    if kind=="dsa":return sum(min(config.sparse_topk,i+1) for i in range(start,stop))
    if kind=="qsa":
        size=config.compression_size
        return sum(min(config.sparse_topk,(i+1)//size*size)+(i+1)%size for i in range(start,stop))
    if kind=="nsa" and branch==1:
        return sum(min(config.local_window,i+1) for i in range(start,stop))
    size=config.compression_size if kind in ("qsa","nsa") else config.block_size
    if kind=="moba":top=max(0,config.sparse_topk//size-1)
    else:top=max(1,(config.sparse_topk+size-1)//size)
    return sum(min(top,i//size)*size+i%size+1 for i in range(start,stop))


def tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,(list,tuple)):
        for item in value:yield from tensors(item)
    elif isinstance(value,dict):
        for item in value.values():yield from tensors(item)


class Ledger(TorchDispatchMode):
    ZERO={"_to_copy","_unsafe_view","alias","arange","as_strided","as_strided_",
          "cat","clone","constant_pad_nd","contiguous","copy_","detach","detach_",
          "empty","empty_like","empty_strided","expand","fill_","flip","full","full_like",
          "gather","index","index_put","index_put_","lift_fresh","masked_fill","masked_fill_",
          "new_empty","new_empty_strided","new_zeros","ones","ones_like","permute","randn",
          "randint","scalar_tensor","select","select_backward","slice","slice_backward",
          "split","split_with_sizes","squeeze","stack","t","transpose","unbind","uniform_",
          "unsqueeze","view","view_as","where","zero_","zeros","zeros_like",
          "any","all","eq","ne","lt","le","gt","ge","bitwise_and","bitwise_or",
          "bitwise_not","topk","floor_divide","remainder","isfinite","normal_",
          "tril","tril_","triu","triu_","eye"}
    # Exact rules already restricted to the triangle; never pair-rescaled.
    PAIR_EXACT={"linalg_solve_triangular"}
    ELEMENT={"add":1,"add_":1,"sub":1,"sub_":1,"rsub":1,"mul":1,"mul_":1,
             "div":1,"div_":1,"neg":1,"exp":1,"log":1,"sqrt":1,"rsqrt":1,"reciprocal":1,
             "sin":1,"cos":1,"tanh":1,"clamp_min":1,"clamp_max":1,"clamp":1,
             "relu":1,"abs":1,"sign":1,"signbit":1,"pow":1,"sigmoid":4,
             "silu":5,"softplus":4,"elu":3,"lerp":3,"lerp_":3,
             "sigmoid_backward":3,"tanh_backward":3,"threshold_backward":1,
             "silu_backward":8,"softplus_backward":5,"elu_backward":3}

    def __init__(self,pair_block=None):
        super().__init__();self.phase="forward"
        self.ops=defaultdict(lambda:dict(calls=0,theory=0,padded_envelope=0,matrix=0))
        self.unknown=Counter();self.scalars_ignored=0
        # Chunk length C of a chunked linear backend; C x C tiles are causal-pair charged.
        self.pair_block=pair_block

    def _pair(self,v):
        c=self.pair_block
        return c is not None and v.dim()>=2 and tuple(v.shape[-2:])==(c,c)

    def _causal_pairs(self,v):
        """Element count of a C x C tile stack restricted to i >= j."""
        c=self.pair_block
        return v.numel()//(c*c)*(c*(c+1)//2) if self._pair(v) else v.numel()

    def add(self,name,count,matrix=0,padded=None):
        item=self.ops[self.phase+"/"+name]
        item["calls"]+=1;item["theory"]+=int(count);item["matrix"]+=int(matrix)
        item["padded_envelope"]+=int(count if padded is None else padded)

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};name=func._schema.name.split("::")[-1]
        # Only diagnostic scalar conversions occur in the counted forward.
        # Routing depths are supplied explicitly, never inferred from fake data.
        if name=="_local_scalar_dense":self.scalars_ignored+=1;return 0
        result=func(*args,**kwargs)
        ins=list(tensors(args));outs=list(tensors(result))
        floats=[v for v in outs if v.is_floating_point()]
        if name in self.ZERO or not floats:return result
        n=sum(v.numel() for v in floats);cost=0;matrix=0
        if name in ("mm","bmm","addmm","baddbmm"):
            a,b=args[-2:] if name in ("mm","bmm") else args[1:3]
            matrix=2*math.prod(a.shape[:-2])*a.shape[-2]*a.shape[-1]*b.shape[-1]
            cost=matrix+(outs[0].numel() if name in ("addmm","baddbmm") else 0)
        elif name in self.ELEMENT:
            cost=n*self.ELEMENT[name]
            if name in ("add","add_","sub","sub_") and kwargs.get("alpha",1)!=1:cost+=n
        elif name in ("sum","mean","amax"):
            cost=max(0,args[0].numel()-outs[0].numel())
            if name=="mean":cost+=outs[0].numel()
        elif name in ("_softmax","_log_softmax"):
            cost=5*args[0].numel()
        elif name in ("_softmax_backward_data","_log_softmax_backward_data"):
            cost=4*args[0].numel()
        elif name=="log_sigmoid_forward":cost=4*args[0].numel()
        elif name=="log_sigmoid_backward":cost=5*args[0].numel()
        elif name=="linalg_vector_norm":cost=2*args[0].numel()+outs[0].numel()
        elif name=="embedding":cost=0
        elif name=="embedding_dense_backward":cost=args[0].numel()
        elif name in ("scatter_add","scatter_add_"):cost=args[3].numel()
        elif name=="native_layer_norm":cost=7*args[0].numel()+5*outs[1].numel()
        elif name=="native_layer_norm_backward":
            # dX plus optional affine reductions; shared unit-op convention.
            cost=(10*args[1].numel() if args[-1][0] else 0)
            cost+=sum(2*args[1].numel() for active in args[-1][1:] if active)
        elif name=="nll_loss_forward":
            cost=2*args[1].numel()+(1 if args[3]==1 else 0)
        elif name=="nll_loss_backward":cost=2*args[2].numel()
        elif name=="binary_cross_entropy_with_logits":cost=10*args[0].numel()
        elif name=="convolution":
            weight=args[1];matrix=2*outs[0].numel()*math.prod(weight.shape[1:]);cost=matrix
            if args[2] is not None:cost+=outs[0].numel()
        elif name=="convolution_backward":
            dy,x,weight=args[:3];mask=args[-1]
            one=2*dy.numel()*math.prod(weight.shape[1:])
            matrix=one*(int(mask[0])+int(mask[1]));cost=matrix+(dy.numel() if mask[2] else 0)
        elif name=="cumsum":
            length=args[0].shape[args[1]] if args[0].dim() else 1
            cost=args[0].numel()-args[0].numel()//max(length,1)
        elif name=="linalg_solve_triangular":
            # left=True: A [*,n,n], B [*,n,R]; per column n(n-1) mul/sub, +n div if not unit.
            a,rhs=args[:2];n=a.shape[-1];columns=outs[0].numel()//n
            matrix=columns*n*(n-1);cost=matrix+(0 if kwargs.get("unitriangular",False) else columns*n)
        else:
            self.unknown[str(func)]+=1
        padded=None
        if cost and name not in self.PAIR_EXACT and any(self._pair(v) for v in ins+outs):
            # Intra-chunk tile: charge causal pairs; the dense tile is padded envelope.
            padded=cost;c=self.pair_block;pairs=c*(c+1)//2
            if name in ("sum","mean","amax"):
                inputs,outputs=self._causal_pairs(args[0]),self._causal_pairs(outs[0])
                cost=max(0,inputs-outputs)+(outputs if name=="mean" else 0);matrix=0
            else:
                if (cost*pairs)%(c*c) or (matrix*pairs)%(c*c):
                    raise RuntimeError(f"non-tile operator on a chunk tile: {func}")
                cost=cost*pairs//(c*c);matrix=matrix*pairs//(c*c)
        if cost:self.add(str(func),cost,matrix,padded)
        return result

    def report(self):
        totals={phase:sum(v["theory"] for k,v in self.ops.items() if k.startswith(phase+"/"))
                for phase in ("forward","backward","optimizer")}
        return dict(formula_version=FORMULA_VERSION,phases=totals,pair_block=self.pair_block,
                    total=sum(totals.values()),matrix_total=sum(v["matrix"] for v in self.ops.values()),
                    padded_envelope=sum(v["padded_envelope"] for v in self.ops.values()),
                    operators=dict(sorted(self.ops.items())),unknown=dict(self.unknown),
                    diagnostic_scalar_reads=self.scalars_ignored)


class _AttentionCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,edges,padded,ledger,label):
        ctx.save_for_backward(q,k,v);ctx.cost=(edges,padded,ledger,label)
        d=q.shape[-1];vd=v.shape[-1]
        matrix=2*edges*(d+vd);total=matrix+6*edges
        ledger.add(label,total,matrix,2*padded*(d+vd)+6*padded)
        return q.new_empty(*q.shape[:-1],vd)

    @staticmethod
    def backward(ctx,grad):
        q,k,v=ctx.saved_tensors;edges,padded,ledger,label=ctx.cost
        needq,needk,needv=ctx.needs_input_grad[:3]
        d=q.shape[-1];vd=v.shape[-1]
        factor=2*d*(int(needq)+int(needk))+2*vd*(int(needq or needk)+int(needv))
        soft=5 if needq or needk else 0
        ledger.add(label,edges*(factor+soft),edges*factor,padded*(factor+soft))
        return (torch.empty_like(q) if needq else None,torch.empty_like(k) if needk else None,
                torch.empty_like(v) if needv else None,None,None,None,None)


@contextmanager
def attention_rules(model,ledger,stage):
    """Replace only numerics in the shape-only cost run, restore on all exits."""
    original=flash_attention.flash_attn.flash_attn_func
    original_indexed=sparse.indexed_attention
    active={};handles=[]
    def enter(module,args):
        active.update(config=module.config,kind=module.kind,sequence=args[0].shape[1],calls=0)
    for module in model.modules():
        if isinstance(module,sparse.SparseMixer):handles.append(module.register_forward_pre_hook(enter))
    def native(q,k,v,causal=False,window_size=(-1,-1)):
        b,t,h,d=q.shape
        edges=b*h*(causal_edges(t,window_size[0]) if causal else t*t)
        # CPU SDPA mask/padding plus flash score recomputation are an envelope,
        # not a measured instruction count. Core theory excludes kernel replay.
        return _AttentionCost.apply(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                 edges,b*h*t*t,ledger,"native_attention").transpose(1,2)
    def indexed(q,k,v,indices,valid):
        c=active["config"];calls=active["calls"];active["calls"]+=1
        branches=2 if active["kind"]=="nsa" else 1
        start=(calls//branches)*c.reference_chunk;stop=start+q.shape[-2]
        edges=q.shape[0]*q.shape[1]*selected_edges(c,start,stop,stage,calls%branches,kind=active["kind"])
        return _AttentionCost.apply(q,k,v,edges,indices.numel(),ledger,"indexed_attention")
    flash_attention.flash_attn.flash_attn_func=native;sparse.indexed_attention=indexed
    try:yield
    finally:
        flash_attention.flash_attn.flash_attn_func=original;sparse.indexed_attention=original_indexed
        for handle in handles:handle.remove()


def count_optimizer(model,ledger):
    """One logical optimizer update; no communication or unused padded ranks."""
    adam=getattr(adamw_step_fused,"_torchdynamo_orig_callable",adamw_step_fused)
    muon=getattr(muon_step_fused,"_torchdynamo_orig_callable",muon_step_fused)
    scalar=lambda value:torch.tensor(float(value))
    ledger.phase="optimizer"
    with torch.no_grad(),ledger:
        for group in parameter_groups(model):
            if group["kind"]=="adamw":
                for p in group["params"]:
                    adam(p,torch.empty_like(p),torch.empty_like(p),torch.empty_like(p),
                         *[scalar(x) for x in (1,group["lr"],*group["betas"],group["eps"],group["weight_decay"])])
            else:
                params=group["params"];m,n=params[0].shape;shape=(len(params),m,n)
                stack=torch.empty(shape);red=-1 if m>=n else -2
                second=torch.empty((len(params),m,1) if red==-1 else (len(params),1,n))
                muon(torch.empty_like(stack),stack,torch.empty_like(stack),second,
                     *[scalar(x) for x in (group["momentum"],group["lr"],group["weight_decay"],group["beta2"])],
                     group["ns_steps"],red)


def audit_graph(config,*,sequence_len=None,batch=1,stage="joint",depth=None,optimizer=True):
    """A complete fake execution, with no real tensor allocation or GPU use."""
    t=sequence_len or config.sequence_len
    pair=CHUNK_BACKENDS.get(config.backend)
    if pair in (config.width,config.heads,config.head_dim,config.mlp_width,
                config.vocab_size,config.gate_rank,config.conv_size):
        raise ValueError("a model dimension equals the chunk length; C x C tiles would be ambiguous")
    ledger=Ledger(pair)
    with FakeTensorMode(allow_non_fake_inputs=True):
        # Initialization is outside the cost scope (one-off setup, not training).
        model=build_model(config,"cpu")
        if config.method in PLUGIN_METHODS:step=stage_start(config,stage)
        else:step=0 if stage=="burnin" else Protocol().steps-config.ouro_gate_steps if stage=="gate_only" else config.burnin_steps+1
        model.set_training_stage(step)
        if config.method=="huginn":
            if depth is None:raise ValueError("Huginn cost requires an explicit sampled depth")
            model.accounting_depth=depth
        x=torch.zeros((batch,t),dtype=torch.long)
        with attention_rules(model,ledger,stage),ledger:
            loss=model(x,x)
            ledger.phase="backward";loss.backward()
        if optimizer:count_optimizer(model,ledger)
        parameters=sum(p.numel() for p in model.parameters())
    report=ledger.report()
    report.update(method=config.method,model_hash=config.fingerprint,sequence_len=t,batch=batch,
                  stage=stage,depth=depth,parameters=parameters)
    return report
