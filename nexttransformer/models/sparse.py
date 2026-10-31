"""Causal sparse selection with two execution paths for the same function.

`reference` (default): genuinely gathered attention with a custom first-order backward that saves
probabilities, not expanded KV copies; the dense teacher is an explicit fp32 matmul. This is the path the
cost audit bills and the CPU tests check. `kernel` (CUDA, `SparseMixer.backend`): the same selections and
teacher computed by `sparse_kernels` (block-sparse attention over the selected keys, fused teacher).
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

from nexttransformer.models.backbone import CausalSelfAttention, Linear, apply_rotary_emb, norm
from nexttransformer.models.sparse_kernels import (TILE, block_sparse_attention, head_mean_teacher,
                                                 require_kernel_backend, selected_block_attention)

BACKENDS = ("reference", "kernel")


def safe_softmax(scores, valid):
    scores = scores.masked_fill(~valid, -torch.inf)
    nonempty = valid.any(-1, keepdim=True)
    scores = torch.where(nonempty, scores, torch.zeros_like(scores))
    return scores.softmax(-1)*valid.to(scores.dtype)


def gather_keys(x, indices):
    # x B,H,T,D and indices B,H,Q,K; expanded view does not allocate.
    b, h, _, d = x.shape
    q, k = indices.shape[-2:]
    return x[:, :, None].expand(-1, -1, q, -1, -1).gather(
        3, indices[..., None].expand(b, h, q, k, d))


class _IndexedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, valid):
        # q/k/v B,H,T,D; q may be a query chunk of a larger sequence.
        kk, vv = gather_keys(k, indices), gather_keys(v, indices)
        scale = q.shape[-1]**-.5
        calc = torch.float64 if q.dtype == torch.float64 else torch.float32
        scores = (q.to(calc).unsqueeze(-2)*kk.to(calc)).sum(-1)*scale
        p = safe_softmax(scores, valid)
        y = (p[..., None]*vv.to(calc)).sum(-2)
        ctx.save_for_backward(q, k, v, indices, p)
        ctx.scale = scale
        return y.to(q.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        q, k, v, indices, p = ctx.saved_tensors
        calc = p.dtype
        kk, vv = gather_keys(k, indices).to(calc), gather_keys(v, indices).to(calc)
        g = dy.to(calc).unsqueeze(-2)
        dp = (g*vv).sum(-1)
        ds = p*(dp-(dp*p).sum(-1, keepdim=True))*ctx.scale
        dq = (ds[..., None]*kk).sum(-2)
        dk_selected = ds[..., None]*q.to(calc).unsqueeze(-2)
        dv_selected = p[..., None]*g
        b, h, t, d = k.shape
        flat = indices.reshape(b, h, -1, 1)
        dk = torch.zeros_like(k, dtype=calc).scatter_add(
            2, flat.expand(-1, -1, -1, d), dk_selected.reshape(b, h, -1, d))
        vd = v.shape[-1]
        dv = torch.zeros_like(v, dtype=calc).scatter_add(
            2, flat.expand(-1, -1, -1, vd), dv_selected.reshape(b, h, -1, vd))
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def indexed_attention(q, k, v, indices, valid):
    return _IndexedAttention.apply(q, k, v, indices, valid)


def indices_to_mask(indices, valid, token_count):
    """Selected-token set as a boolean mask. int32 scatter_add: clamped duplicate indices cannot overwrite a True."""
    m = torch.zeros(*indices.shape[:-1], token_count, dtype=torch.int32, device=indices.device)
    m.scatter_add_(-1, indices, valid.to(torch.int32))
    return m > 0


def select_tokens(scores, allowed, k):
    count = min(k, scores.shape[-1])
    chosen = scores.masked_fill(~allowed, -torch.inf).topk(count, dim=-1).indices
    return chosen, allowed.expand_as(scores).gather(-1, chosen)


def block_indices(scores, full_visible, query_positions, block_size, token_count, topk):
    """Select complete historical blocks and force the current partial block.

    topk counts historical blocks here; the forced tail is separately accounted.
    All expanded token positions are deduplicated by excluding current block.
    """
    b, h, q, blocks = scores.shape
    current = query_positions//block_size
    eligible = full_visible & (torch.arange(blocks, device=scores.device)[None, :] != current[:, None])
    chosen, ok = select_tokens(scores, eligible[None, None], topk)
    offsets = torch.arange(block_size, device=scores.device)
    indices = (chosen[..., None]*block_size+offsets).flatten(-2)
    valid = ok[..., None].expand(-1, -1, -1, -1, block_size).flatten(-2)
    tail = (current[:, None]*block_size+offsets)[None, None].expand(b, h, -1, -1)
    indices = torch.cat((indices, tail), -1)
    valid = torch.cat((valid, torch.ones_like(tail, dtype=torch.bool)), -1)
    valid = valid & (indices <= query_positions[None, None, :, None]) & (indices < token_count)
    return indices.clamp_max(token_count-1), valid


def qsa_indices(scores,complete,positions,size,token_count,token_budget):
    """QSA Eq. 16/19: complete blocks, then the FINAL INCOMPLETE tail.

    At a block boundary the just-completed block competes in Top-K and there
    is no forced tail. Unlike MoBA, it is not always a forced current block.
    """
    top=max(1,(token_budget+size-1)//size)
    chosen,ok=select_tokens(scores,complete[None,None],top)
    offsets=torch.arange(size,device=scores.device)
    indices=(chosen[...,None]*size+offsets).flatten(-2)[...,:token_budget]
    valid=ok[...,None].expand(*ok.shape,size).flatten(-2)[...,:token_budget]
    tail=((positions+1)//size*size)[:,None]+offsets
    tail=tail[None,None].expand(*scores.shape[:3],size)
    tail_valid=(tail<=positions[None,None,:,None])&(tail<token_count)
    return (torch.cat((indices,tail),-1).clamp_max(token_count-1),
            torch.cat((valid,tail_valid),-1))


def pool_teacher(teacher,size,blocks,kind):
    grouped=F.pad(teacher,(0,blocks*size-teacher.shape[-1])).reshape(*teacher.shape[:-1],blocks,size)
    return grouped.amax(-1) if kind=="qsa" else grouped.sum(-1)


def pool_blocks(x, size):
    b, h, t, d = x.shape
    blocks = (t+size-1)//size
    padded = F.pad(x, (0, 0, 0, blocks*size-t))
    return padded.reshape(b, h, blocks, size, d).mean(-2)


def partial_rope(x, positions, rotary_dim):
    if not rotary_dim: return x
    inv = 100000.**(-torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32)/rotary_dim)
    angle = positions.float()[:, None]*inv[None]
    cos, sin = angle.cos().to(x)[None, :, None], angle.sin().to(x)[None, :, None]
    rotary = apply_rotary_emb(x[..., :rotary_dim], cos, sin)
    return torch.cat((rotary, x[..., rotary_dim:]), -1)


class SparseMixer(CausalSelfAttention):
    def __init__(self, config, layer_idx, kind=None):
        super().__init__(config, layer_idx)
        self.config = config
        self.kind = kind or config.method
        self.in_burnin = True   # set by GPT.set_step; a bool, not the step, so compiled graphs change only at the switch
        self.backend = "reference"
        self.last_aux = None
        self.last_stats = {}
        d = config.width
        if self.kind in ("dsa", "qsa", "seer"):
            self.index_q = Linear(d, config.index_heads*config.index_dim, bias=False)
            self.index_k = Linear(d, config.index_dim, bias=False)
            if self.kind == "dsa":
                self.index_weights = Linear(d, config.index_heads, bias=False)
                self.index_norm = nn.LayerNorm(config.index_dim)
        if self.kind == "nsa":
            size, hd = config.compression_size, config.head_dim
            self.compress_k = Linear(size*hd, hd, bias=False)
            self.compress_v = Linear(size*hd, hd, bias=False)
            self.compress_position = nn.Parameter(torch.zeros(size, hd))
            # NSA: gate scores g^c_t come from an MLP + sigmoid on the input; the official FLA
            # implementation emits one gate per head and per branch (g_proj: hidden -> heads*3).
            self.branch_gate = Linear(d, 3*config.heads, bias=False)

    def _projections(self, x, ve, cos_sin):
        b, t, _ = x.shape
        q, k, v = (f(x).view(b, t, self.n_head, self.head_dim)
                   for f in (self.c_q, self.c_k, self.c_v))
        if ve is not None:
            gate = 3*torch.sigmoid(self.ve_gate(x[..., :12]))
            v = v+gate.unsqueeze(-1)*ve.view_as(v)
        q, k = (norm(apply_rotary_emb(z, *cos_sin))*1.2 for z in (q, k))
        return tuple(z.transpose(1, 2) for z in (q, k, v))

    def _index(self, x):
        c = self.config
        # The indexer is trained by its distillation loss only; the backbone gets no gradient from it.
        # DSA (DeepSeek-V3.2 report): "we detach the indexer input from the computational graph ...";
        # SeerAttention trains its gates on a frozen backbone; QSA does not say, so it follows the other two.
        x = x.detach()
        b, t, _ = x.shape
        iq = self.index_q(x).reshape(b, t, c.index_heads, c.index_dim)
        ik = self.index_k(x).reshape(b, t, 1, c.index_dim)
        size = c.compression_size if self.kind == "qsa" else c.block_size if self.kind == "seer" else 1
        if size > 1:
            ik = pool_blocks(ik.transpose(1, 2), size).transpose(1, 2)
        if self.kind == "dsa":
            ik = self.index_norm(ik.float()).to(x.dtype)
        else:
            iq, ik = norm(iq), norm(ik)
        iq = partial_rope(iq, torch.arange(t,device=x.device), c.index_rope_dim)
        ik = partial_rope(ik, torch.arange(ik.shape[1],device=x.device)*size, c.index_rope_dim)
        weights = self.index_weights(x).transpose(1, 2) if self.kind == "dsa" else None
        return iq.transpose(1, 2), ik.transpose(1, 2), weights, size

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None: raise NotImplementedError("sparse reference does not use native KV cache")
        c = self.config
        b, t, d = x.shape
        q, k, v = self._projections(x, ve, cos_sin)
        learned = self.kind in ("dsa", "qsa", "seer")
        burnin = learned and self.training and self.in_burnin
        aux = []
        outputs = []
        edges, teacher_edges = 0, 0
        if self.backend not in BACKENDS: raise ValueError(f"unknown sparse backend {self.backend!r}")
        kernel = self.backend == "kernel"
        if kernel: require_kernel_backend(q, t)
        chunk = t if kernel else c.reference_chunk
        if learned: iq, ik, iw, size = self._index(x)
        else: size = c.compression_size if self.kind == "nsa" else c.block_size
        if self.kind == "moba": pooled = pool_blocks(k, size)
        if self.kind == "nsa":
            count = (t+size-1)//size
            kp = F.pad(k,(0,0,0,count*size-t)).reshape(b,c.heads,count,size,c.head_dim)
            vp = F.pad(v,(0,0,0,count*size-t)).reshape_as(kp)
            ck = self.compress_k((kp+self.compress_position.to(kp)).flatten(-2))
            cv = self.compress_v(vp.flatten(-2))
            gates = self.branch_gate(x).sigmoid().view(b,t,c.heads,3)
        for start in range(0, t, chunk):
            stop = min(start+chunk,t)
            positions = torch.arange(start,stop,device=x.device)
            qq = q[:, :, start:stop]
            causal = torch.arange(t,device=x.device)[None,:] <= positions[:,None]
            if learned:
                scores = torch.einsum("bhqd,bhkd->bhqk",iq[:,:,start:stop],ik.expand(-1,c.index_heads,-1,-1))
                scores = scores.relu()
                if iw is not None: scores = scores*iw[:,:,start:stop,None]*(c.index_heads**-.5)
                scores = scores.sum(1)
                if self.kind!="qsa":scores=scores/c.index_dim**.5
                complete = (torch.arange(ik.shape[2],device=x.device)+1)*size-1 <= positions[:,None]
                if self.kind == "dsa":
                    indices, valid = select_tokens(scores[:,None],causal[None,None],c.sparse_topk)
                elif self.kind == "qsa":
                    indices,valid=qsa_indices(scores[:,None],complete,positions,size,t,c.sparse_topk)
                else:
                    indices, valid = block_indices(scores[:,None],complete,positions,size,t,
                                                   max(1,(c.sparse_topk+size-1)//size))
                if kernel: sel_mask = indices_to_mask(indices, valid, t)   # B,1,Q,T, shared by all attention heads
                indices,valid=(a.expand(-1,c.heads,-1,-1) for a in (indices,valid))
                if self.training:
                    # Self-distillation is explicitly an additional dense QK path.
                    with torch.no_grad():
                        domain = causal[None,None].expand(b,c.heads,-1,-1)
                        if not burnin:
                            if kernel: domain = sel_mask.expand(-1,c.heads,-1,-1)
                            else:
                                selected_domain = torch.zeros_like(domain,dtype=torch.int32)
                                selected_domain.scatter_add_(-1,indices,valid.to(torch.int32))
                                domain = selected_domain>0
                        if kernel:
                            teacher = head_mean_teacher(qq,k,None if burnin else sel_mask[:,0],start,c.head_dim**-.5)
                        else:
                            raw = torch.matmul(qq.float(),k.float().transpose(-2,-1))*c.head_dim**-.5
                            teacher = safe_softmax(raw,domain).mean(1)
                        student_domain=complete[None].expand(b,-1,-1)
                        if size > 1:
                            teacher=pool_teacher(teacher,size,ik.shape[2],self.kind)
                            if not burnin:
                                block_domain=F.pad(domain[:,0],(0,ik.shape[2]*size-t)).reshape(b,stop-start,-1,size).any(-1)
                                student_domain=student_domain&block_domain
                        elif not burnin:student_domain=student_domain&domain[:,0]
                        teacher = teacher*complete.to(teacher.dtype)
                        teacher = teacher/teacher.sum(-1,keepdim=True).clamp_min(1e-12)
                    student = safe_softmax(scores.float(),student_domain)
                    loss = (teacher*(teacher.clamp_min(1e-12).log()-student.clamp_min(1e-12).log())).sum(-1)
                    aux.append(loss.sum()/(b*t))
                    teacher_edges += b*c.heads*(stop-start)*t
                if burnin:
                    indices = torch.arange(t,device=x.device)[None,None,None].expand(b,c.heads,stop-start,-1)
                    valid = causal[None,None].expand_as(indices)
                if kernel: out = block_sparse_attention(qq,k,v,causal[None,None] if burnin else sel_mask)
                else: out = indexed_attention(qq,k,v,indices,valid)
            elif self.kind == "moba":
                scores = torch.matmul(qq,pooled.transpose(-2,-1))
                complete = (torch.arange(pooled.shape[2],device=x.device)+1)*size-1 <= positions[:,None]
                # K includes the forced current block for MoBA (unlike indexer K).
                if kernel and size == TILE:
                    eligible = complete & (torch.arange(pooled.shape[2],device=x.device)[None,:] != (positions//size)[:,None])
                    chosen,allowed = select_tokens(scores,eligible[None,None],max(0,c.sparse_topk//size-1))
                    out,count = selected_block_attention(qq,k,v,chosen,allowed,positions)
                    edges = edges+count
                    outputs.append(out)
                    continue
                indices,valid=block_indices(scores,complete,positions,size,t,max(0,c.sparse_topk//size-1))
                out=(block_sparse_attention(qq,k,v,indices_to_mask(indices,valid,t)) if kernel
                     else indexed_attention(qq,k,v,indices,valid))
            else:
                complete = (torch.arange(ck.shape[2],device=x.device)+1)*size-1 <= positions[:,None]
                scores=torch.matmul(qq,ck.transpose(-2,-1))*c.head_dim**-.5
                probability=safe_softmax(scores,complete[None,None])
                compressed=torch.matmul(probability,cv)
                # Non-overlapping compression/selection of equal size: identity
                # mapping of compressed importance to selected block importance.
                indices,valid=block_indices(probability,complete,positions,size,t,max(1,c.sparse_topk//size))
                local=(positions[:,None]-torch.arange(c.local_window,device=x.device)[None,:]).flip(-1)
                lv=(local>=0)[None,None].expand(b,c.heads,-1,-1)
                if kernel:
                    kpos=torch.arange(t,device=x.device)[None,:]
                    window=(kpos<=positions[:,None])&(kpos>positions[:,None]-c.local_window)
                    selected=block_sparse_attention(qq,k,v,indices_to_mask(indices,valid,t))
                    local_out=block_sparse_attention(qq,k,v,window[None,None])
                else:
                    selected=indexed_attention(qq,k,v,indices,valid)
                    li=local.clamp_min(0)[None,None].expand_as(lv)
                    local_out=indexed_attention(qq,k,v,li,lv)
                gate=gates[:,start:stop].transpose(1,2)  # B,H,chunk,3: per-head, per-branch
                out=(compressed*gate[...,0:1]+selected*gate[...,1:2]+local_out*gate[...,2:3])
                edges = edges + (lv.sum() if kernel else int(lv.sum()))+b*c.heads*(stop-start)*ck.shape[2]
            edges = edges + (valid.sum() if kernel else int(valid.sum()))   # kernel: no host sync
            outputs.append(out)
        y=torch.cat(outputs,2).transpose(1,2).reshape(b,t,d)
        self.last_aux=sum(aux,x.new_zeros(()))*c.index_loss_weight
        self.last_stats={"selected_edges":edges,"teacher_dense_score_edges":teacher_edges,
                         "burnin":burnin,"backend":self.backend}
        return self.c_proj(y)
