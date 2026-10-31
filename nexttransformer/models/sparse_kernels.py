"""CUDA execution kernels for `sparse.SparseMixer(backend="kernel")`.

They compute the same functions as the reference path in `sparse.py` (same selected key sets, same
distillation teacher); only execution differs. The cost audit bills the reference path.
"""
import functools

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs; the kernel backend then refuses to start
    triton = None

TILE = 128
TEACHER_HEAD_DIMS = (16, 32, 64, 128, 256)


def require_kernel_backend(q, token_count):
    """Raise instead of silently running something else when the kernel backend cannot serve these inputs."""
    problems = []
    if not q.is_cuda: problems.append("tensors are not on CUDA")
    if triton is None: problems.append("triton is not installed")
    if token_count % TILE: problems.append(f"sequence length {token_count} is not a multiple of {TILE}")
    if q.shape[-1] not in TEACHER_HEAD_DIMS: problems.append(f"head_dim {q.shape[-1]} not in {TEACHER_HEAD_DIMS}")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32): problems.append(f"dtype {q.dtype}")
    if problems: raise ValueError("sparse kernel backend unavailable: " + "; ".join(problems))


@functools.cache
def _flex_kernel():
    return torch.compile(flex_attention, dynamic=False)


if triton is not None:
    @triton.jit
    def _teacher_lse_kernel(Q, K, M, LSE, sqb, sqh, sqm, sqd, skb, skh, skn, skd, smb, smm, smn, slb, slh, slm,
                            H, QLEN, KLEN, q_start, scale, HAS_MASK: tl.constexpr, PRECISION: tl.constexpr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
        pid_m, pid_bh = tl.program_id(0), tl.program_id(1)
        b, h = pid_bh // H, pid_bh % H
        offs_m = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        row_ok = offs_m < QLEN
        pos = q_start + offs_m
        q = tl.load(Q + b*sqb + h*sqh + offs_m[:, None]*sqm + offs_d[None, :]*sqd, mask=row_ok[:, None], other=0.)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        for n in range(0, tl.minimum(KLEN, q_start + (pid_m+1)*BLOCK_M), BLOCK_N):
            offs_n = n + tl.arange(0, BLOCK_N)
            col_ok = offs_n < KLEN
            k = tl.load(K + b*skb + h*skh + offs_n[:, None]*skn + offs_d[None, :]*skd, mask=col_ok[:, None], other=0.)
            s = tl.dot(q, tl.trans(k), input_precision=PRECISION)*scale
            valid = row_ok[:, None] & col_ok[None, :] & (offs_n[None, :] <= pos[:, None])
            if HAS_MASK:
                valid = valid & (tl.load(M + b*smb + offs_m[:, None]*smm + offs_n[None, :]*smn, mask=valid, other=0) != 0)
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0., m_new)
            l_i = l_i*tl.exp(m_i - m_safe) + tl.sum(tl.exp(s - m_safe[:, None]), 1)
            m_i = m_new
        # empty domain -> +inf, so the second pass gives exp(s - lse) = 0 like safe_softmax
        tl.store(LSE + b*slb + h*slh + offs_m*slm, tl.where(l_i > 0, m_i + tl.log(l_i), float("inf")), mask=row_ok)

    @triton.jit
    def _teacher_mean_kernel(Q, K, M, LSE, OUT, sqb, sqh, sqm, sqd, skb, skh, skn, skd, smb, smm, smn, slb, slh, slm,
                             sob, som, son, H, QLEN, KLEN, q_start, scale, HAS_MASK: tl.constexpr,
                             PRECISION: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
        pid_m, pid_n, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        if pid_n*BLOCK_N > q_start + (pid_m+1)*BLOCK_M - 1:
            return  # tile entirely above the causal diagonal; OUT is pre-zeroed
        offs_m = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        row_ok, col_ok = offs_m < QLEN, offs_n < KLEN
        valid = row_ok[:, None] & col_ok[None, :] & (offs_n[None, :] <= (q_start + offs_m)[:, None])
        if HAS_MASK:
            valid = valid & (tl.load(M + b*smb + offs_m[:, None]*smm + offs_n[None, :]*smn, mask=valid, other=0) != 0)
        acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for h in range(0, H):
            q = tl.load(Q + b*sqb + h*sqh + offs_m[:, None]*sqm + offs_d[None, :]*sqd, mask=row_ok[:, None], other=0.)
            k = tl.load(K + b*skb + h*skh + offs_n[:, None]*skn + offs_d[None, :]*skd, mask=col_ok[:, None], other=0.)
            lse = tl.load(LSE + b*slb + h*slh + offs_m*slm, mask=row_ok, other=float("inf"))
            s = tl.dot(q, tl.trans(k), input_precision=PRECISION)*scale
            acc += tl.where(valid, tl.exp(s - lse[:, None]), 0.)
        tl.store(OUT + b*sob + offs_m[:, None]*som + offs_n[None, :]*son, acc/H, mask=row_ok[:, None] & col_ok[None, :])


def head_mean_teacher(q, k, mask, q_start, scale, block=64):
    """`safe_softmax(q k^T * scale, domain).mean(1)` without materialising the B,H,Q,T score tensor.

    Two Triton passes: a per-head row log-sum-exp, then the head-summed probabilities tile by tile. Scores are
    exact products accumulated in float32 (bf16 tensor-core inputs, or IEEE float32), as in the float32 matmul
    path. q: B,H,Q,D for absolute
    positions q_start..; k: B,H,T,D; mask: B,Q,T bool shared by all heads, or None for the causal burn-in domain.
    The domain must be causal (every selection is); keys after the query are never read. Returns B,Q,T float32.
    """
    b, h, qlen, d = q.shape
    klen = k.shape[2]
    lse = torch.empty(b, h, qlen, device=q.device, dtype=torch.float32)
    out = torch.zeros(b, qlen, klen, device=q.device, dtype=torch.float32)
    if mask is not None:
        m = mask.to(torch.uint8)  # a cast, not view(dtype): inductor cannot lower a bool -> uint8 reinterpret
        mstride = m.stride()
    else:
        m, mstride = lse, (0, 0, 0)
    meta = dict(HAS_MASK=mask is not None, PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
                BLOCK_M=block, BLOCK_N=block, D=d)
    _teacher_lse_kernel[(triton.cdiv(qlen, block), b*h)](
        q, k, m, lse, *q.stride(), *k.stride(), *mstride, *lse.stride(), h, qlen, klen, q_start, scale, **meta)
    _teacher_mean_kernel[(triton.cdiv(qlen, block), triton.cdiv(klen, block), b)](
        q, k, m, lse, out, *q.stride(), *k.stride(), *mstride, *lse.stride(), *out.stride(),
        h, qlen, klen, q_start, scale, **meta)
    return out


def _tile_lists(selected):
    """Per (batch, head, query tile): how many KV tiles are selected, and their indices first (BlockMask layout)."""
    count = selected.sum(-1, dtype=torch.int32)
    order = torch.argsort(selected.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
    return count, order


@functools.cache
def _metadata_kernel(function):
    return torch.compile(function, dynamic=False, fullgraph=True)


def _pack_block_mask(mask):
    """Pack exactly the requested keys, including the safe key for empty rows."""
    mb, mh, q_len, kv_len = mask.shape
    has_key = mask.any(-1, keepdim=True)
    first = torch.arange(kv_len, device=mask.device) == 0
    mask = mask | (~has_key & first)
    tiles = mask.view(mb, mh, q_len//TILE, TILE, kv_len//TILE, TILE)
    full = tiles.all(dim=5).all(dim=3)
    partial = tiles.any(dim=5).any(dim=3) & ~full
    bits = mask.view(mb, mh, q_len, kv_len//8, 8).to(torch.int32)
    packed = (bits << torch.arange(8, device=mask.device, dtype=torch.int32)).sum(-1, dtype=torch.int32).to(torch.uint8)
    return has_key, full, partial, packed


def _selected_block_layout(chosen, valid, positions, blocks):
    """MoBA layout directly from historical blocks and the mandatory causal tail.

    The small [B,H,Q,KV-block] bitmap replaces the expanded [B,H,Q,KV-token]
    mask. The top-k caller excludes the current block from valid selections.
    """
    b, h, t, _ = chosen.shape
    selected = torch.zeros((b, h, t, blocks), device=chosen.device, dtype=torch.int32)
    selected.scatter_add_(-1, chosen, valid.to(torch.int32))
    current = (positions//TILE).view(1, 1, t, 1).expand(b, h, t, 1)
    selected.scatter_(-1, current, 1)
    selected = selected > 0
    tiles = selected.view(b, h, t//TILE, TILE, blocks)
    before = torch.arange(blocks, device=chosen.device)[None, :] < torch.arange(t//TILE, device=chosen.device)[:, None]
    full = tiles.all(dim=3) & before
    partial = tiles.any(dim=3) & ~full
    edges = valid.sum()*TILE + b*h*((positions % TILE)+1).sum()
    return selected, full, partial, edges


def selected_block_attention(q, k, v, chosen, valid, positions):
    """Full-row causal attention for MoBA's TILE-sized historical blocks."""
    b, h, q_len, _ = q.shape
    if q_len != k.shape[2] or q_len % TILE:
        raise ValueError("selected-block attention requires complete TILE-aligned rows")
    layout = (_selected_block_layout if torch.compiler.is_compiling() or not q.is_cuda
              else _metadata_kernel(_selected_block_layout))
    selected, full, partial, edges = layout(chosen, valid, positions, k.shape[2]//TILE)
    if b > 1 and h > 1:
        def mask_mod(b, h, qi, ki): return selected[b, h, qi, ki//TILE] & (ki <= qi)
    elif b > 1:
        per_batch = selected[:, 0]
        def mask_mod(b, h, qi, ki): return per_batch[b, qi, ki//TILE] & (ki <= qi)
    elif h > 1:
        per_head = selected[0]
        def mask_mod(b, h, qi, ki): return per_head[h, qi, ki//TILE] & (ki <= qi)
    else:
        shared = selected[0, 0]
        def mask_mod(b, h, qi, ki): return shared[qi, ki//TILE] & (ki <= qi)
    block = BlockMask.from_kv_blocks(*_tile_lists(partial), *_tile_lists(full), BLOCK_SIZE=TILE,
                                   mask_mod=mask_mod, seq_lengths=(q_len, k.shape[2]))
    return _flex_kernel()(q, k, v, block_mask=block), edges


def block_sparse_attention(q, k, v, mask):
    """softmax over exactly the masked-in keys through FlexAttention's block-sparse kernel.

    KV tiles with no selected key are skipped, fully selected tiles run unmasked and
    partial tiles apply the exact mask. The tile lists are reduced from the mask itself (not create_block_mask,
    whose compiled path built a wrong backward layout for Q < KV in torch 2.9.1). mask is (1|B, 1|H, Q, T) bool;
    it is bit-packed so the copy kept for backward is 1/8 of a bool mask. Rows with no key get key 0 inside the
    kernel and return 0, as in the reference `safe_softmax`.
    """
    mb, mh, q_len, kv_len = mask.shape
    prepare = (_pack_block_mask if torch.compiler.is_compiling() or not mask.is_cuda
               else _metadata_kernel(_pack_block_mask))
    has_key, full, partial, packed = prepare(mask)

    def bit(byte, ki): return ((byte.to(torch.int32) >> (ki % 8)) & 1) == 1
    if mb > 1 and mh > 1:
        def mask_mod(b, h, qi, ki): return bit(packed[b, h, qi, ki//8], ki)
    elif mb > 1:
        per_batch = packed[:, 0]
        def mask_mod(b, h, qi, ki): return bit(per_batch[b, qi, ki//8], ki)
    elif mh > 1:
        per_head = packed[0]
        def mask_mod(b, h, qi, ki): return bit(per_head[h, qi, ki//8], ki)
    else:
        shared = packed[0, 0]
        def mask_mod(b, h, qi, ki): return bit(shared[qi, ki//8], ki)

    block = BlockMask.from_kv_blocks(*_tile_lists(partial), *_tile_lists(full), BLOCK_SIZE=TILE,
                                     mask_mod=mask_mod, seq_lengths=(q_len, kv_len))
    out = _flex_kernel()(q, k, v, block_mask=block)
    return torch.where(has_key, out, torch.zeros_like(out))
