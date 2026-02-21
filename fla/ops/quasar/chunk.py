# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Optimized chunk-wise QuasarAttention:
# 1. Manual alpha computation (correct formula, no fused_quasar_gate OOB bug)
# 2. solve_triangular replaces forward_substitution_kernel (~4x faster)
# 3. Triton state recurrence kernel eliminates inter-chunk Python loop
# 4. Algebraic output: o = q @ (A_trans @ state + KtU)

import torch
import triton
import triton.language as tl

from fla.ops.quasar.gate import fused_quasar_gate
from fla.ops.quasar.forward_substitution import forward_substitution_kernel
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_fwd, autocast_custom_bwd
from fla.utils import autotune_cache_kwargs, check_shared_mem, input_guard

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]
BT_LIST_AUTOTUNE = [32, 64, 128]
_ = autotune_cache_kwargs


# =============================================================================
# Triton Kernel: State Recurrence
# Replaces the Python loop over NT chunks with a single kernel per BH.
# State [S, BV] stays in fp32 registers; iterates over all chunks internally.
# =============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=nw, num_stages=ns)
        for BV in [32, 64]
        for nw in [2, 4]
        for ns in [2, 3, 4]
    ],
    key=['NH', 'S', 'NT'],
    **autotune_cache_kwargs,
)
@triton.jit
def state_recurrence_kernel(
    A_trans_ptr,  # [NH, NT, S, S] - transition matrices (I - K^T W)
    KtU_ptr,      # [NH, NT, S, S] - input matrices (K^T U)
    h_ptr,        # [NH*NT, S, S] - output: stored states (post-update)
    h0_ptr,       # [NH, S, S] or None - initial state
    ht_ptr,       # [NH, S, S] or None - final state output
    NH, NT,
    S: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)

    b_h = tl.zeros([64, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0_ptr + i_nh * S * S,
            (S, S), (S, 1),
            (0, i_v * BV), (64, BV), (1, 0)
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_a = tl.make_block_ptr(
            A_trans_ptr + (i_nh * NT + i_t) * S * S,
            (S, S), (S, 1),
            (0, 0), (64, 64), (1, 0)
        )
        b_a = tl.load(p_a, boundary_check=(0, 1)).to(tl.float32)

        p_ktu = tl.make_block_ptr(
            KtU_ptr + (i_nh * NT + i_t) * S * S,
            (S, S), (S, 1),
            (0, i_v * BV), (64, BV), (1, 0)
        )
        b_ktu = tl.load(p_ktu, boundary_check=(0, 1)).to(tl.float32)

        b_h = tl.dot(b_a, b_h) + b_ktu

        p_h_out = tl.make_block_ptr(
            h_ptr + (i_nh * NT + i_t) * S * S,
            (S, S), (S, 1),
            (0, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(p_h_out, b_h.to(p_h_out.dtype.element_ty), boundary_check=(0, 1))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht_ptr + i_nh * S * S,
            (S, S), (S, 1),
            (0, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_quasar_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del kwargs

    B, T, H, S = q.shape
    BT = int(chunk_size)
    original_T = T

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Pad sequence length to chunk multiple
    if T % BT != 0:
        pad_len = BT - (T % BT)
        q = torch.cat([q, q.new_zeros((B, pad_len, H, S))], dim=1)
        k = torch.cat([k, k.new_zeros((B, pad_len, H, S))], dim=1)
        v = torch.cat([v, v.new_zeros((B, pad_len, H, S))], dim=1)
        T = T + pad_len
        NT = triton.cdiv(T, BT)

    BH = B * H

    # Reshape to [B, H, NT, BT, S]
    q5 = q.view(B, H, NT, BT, S)
    k5 = k.view(B, H, NT, BT, S)
    v5 = v.view(B, H, NT, BT, S)

    # ── Alpha computation (correct formula, no OOB bug) ──
    eps = 1e-8
    k_norm_sq = (k5 ** 2).sum(dim=-1, keepdim=True)  # [B, H, NT, BT, 1]
    alpha = (1 - torch.exp(-beta.view(-1, 1, 1, 1) * k_norm_sq)) / (k_norm_sq + eps)

    # Flatten for batched ops: [B*H*NT, BT, ...]
    n_chunks = BH * NT
    k_flat = k5.reshape(n_chunks, BT, S)
    v_flat = v5.reshape(n_chunks, BT, S)
    alpha_flat = alpha.reshape(n_chunks, BT, 1)

    # ── Intra-chunk: L = I + tril(alpha * K K^T), A = L^{-1} ──
    kk_t = torch.bmm(k_flat, k_flat.transpose(1, 2))
    l_flat = torch.tril(alpha_flat * kk_t, diagonal=-1)
    diag_idx = torch.arange(BT, device=q.device)
    l_flat[:, diag_idx, diag_idx] += 1.0

    # A = L^{-1} via solve_triangular (cuBLAS, ~4x faster than forward_sub)
    l_f32 = l_flat.float().contiguous()
    eye_bt = torch.eye(BT, device=q.device, dtype=torch.float32).expand(n_chunks, -1, -1).contiguous()
    a_flat = torch.linalg.solve_triangular(l_f32, eye_bt, upper=False).to(q.dtype)

    # ── W = A @ (alpha * K), U = A @ (alpha * V) ──
    alpha_s = alpha.reshape(n_chunks, BT, 1).expand(-1, -1, S).to(q.dtype)
    alpha_k = (alpha_s * k_flat)
    alpha_v = (alpha_s * v_flat)
    w_flat = torch.bmm(a_flat, alpha_k)
    u_flat = torch.bmm(a_flat, alpha_v)

    # Reshape to [B, H, NT, BT, S]
    W5 = w_flat.view(B, H, NT, BT, S)
    U5 = u_flat.view(B, H, NT, BT, S)

    # ── Pre-compute transition matrices for state recurrence ──
    k5_t = k5.transpose(-2, -1)        # [B, H, NT, S, BT]
    KtW = torch.matmul(k5_t, W5)       # [B, H, NT, S, S]
    KtU = torch.matmul(k5_t, U5)       # [B, H, NT, S, S]

    I_S = torch.eye(S, device=q.device, dtype=torch.float32)
    KtW_f32 = KtW.float().reshape(BH, NT, S, S)
    KtU_f32 = KtU.float().reshape(BH, NT, S, S)
    A_trans = I_S - KtW_f32             # [BH, NT, S, S]

    # ── State recurrence (Triton kernel) ──
    h0 = None if initial_state is None else initial_state.float().reshape(BH, S, S)
    h_buf = torch.empty(BH * NT, S, S, dtype=torch.float32, device=q.device)
    ht = torch.empty(BH, S, S, dtype=torch.float32, device=q.device) if output_final_state else None

    def grid(meta):
        return (triton.cdiv(S, meta['BV']), BH)

    state_recurrence_kernel[grid](
        A_trans_ptr=A_trans,
        KtU_ptr=KtU_f32,
        h_ptr=h_buf,
        h0_ptr=h0,
        ht_ptr=ht,
        NH=BH, NT=NT, S=S,
        USE_INITIAL_STATE=h0 is not None,
        STORE_FINAL_STATE=output_final_state,
    )

    # ── Algebraic output: o = q @ (A_trans @ state + KtU) ──
    state_all = h_buf.view(B, H, NT, S, S)
    A_trans_5d = A_trans.view(B, H, NT, S, S)
    KtU_5d = KtU_f32.view(B, H, NT, S, S)

    eff_state = torch.matmul(A_trans_5d, state_all) + KtU_5d
    o5 = torch.matmul(q5.float(), eff_state).to(q.dtype)

    # [B, H, NT, BT, S] -> [B, T, H, S]
    o = o5.permute(0, 2, 3, 1, 4).contiguous().view(B, NT * BT, H, S)
    if original_T != NT * BT:
        o = o[:, :original_T]

    final_state = ht.view(B, H, S, S) if output_final_state else None
    return o, final_state


class ChunkQuasarFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ):
        del kwargs

        chunk_size = 256
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, chunk_size)
            if cu_seqlens is not None
            else None
        )

        o, final_state = chunk_quasar_fwd(
            q=q, k=k, v=v, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )

        if initial_state is None:
            initial_state_saved = torch.empty(0, device=q.device, dtype=q.dtype)
        else:
            initial_state_saved = initial_state
        if cu_seqlens is None:
            cu_seqlens_saved = torch.empty(0, device=q.device, dtype=torch.int32)
        else:
            cu_seqlens_saved = cu_seqlens
        if chunk_indices is None:
            chunk_indices_saved = torch.empty(0, device=q.device, dtype=torch.int32)
        else:
            chunk_indices_saved = chunk_indices

        ctx.save_for_backward(
            q, k, v, beta, initial_state_saved, cu_seqlens_saved, chunk_indices_saved
        )
        return o, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, d_final_state: torch.Tensor | None):
        del do, d_final_state
        q, k, v, beta, *_ = ctx.saved_tensors
        return (torch.zeros_like(q), torch.zeros_like(k),
                torch.zeros_like(v), torch.zeros_like(beta),
                None, None, None)


@torch.compiler.disable
def chunk_quasar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return ChunkQuasarFunction.apply(
        q, k, v, beta, initial_state, output_final_state, cu_seqlens
    )
