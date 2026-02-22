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
# v9: v7 + two-stream overlap (intra on stream 1, recurrence on stream 2).
# Hides Python/launch overhead by overlapping kernel submissions.
# Sequential: 2.046ms → Two streams: 1.582ms (22.7% faster kernel pair).
# =============================================================================
@triton.jit
def intra_chunk_v9(
    K_ptr,          # [B, T, H, S] — original layout
    V_ptr,          # [B, T, H, S]
    beta_ptr,       # [H]
    A_trans_ptr,    # [BH*NT, S, S] output
    KtU_ptr,        # [BH*NT, S, S] output
    T,              # actual T (may not be divisible by BT)
    NT: tl.constexpr,
    BT: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
):
    chunk_id = tl.program_id(0)  # 0..BH*NT-1
    bh = chunk_id // NT
    c = chunk_id % NT
    b = bh // H
    h = bh % H

    si = tl.arange(0, S)
    beta_val = tl.load(beta_ptr + h).to(tl.float32)

    S_KW = tl.zeros((S, S), dtype=tl.float32)
    S_KU = tl.zeros((S, S), dtype=tl.float32)

    # Stride for [B, T, H, S]: dim strides = [T*H*S, H*S, S, 1]
    stride_b = T * H * S
    stride_t = H * S

    for i in range(BT):
        t_idx = c * BT + i
        if t_idx < T:
            row_off = b * stride_b + t_idx * stride_t + h * S + si
            k_i = tl.load(K_ptr + row_off).to(tl.float32)
            v_i = tl.load(V_ptr + row_off).to(tl.float32)

            k_norm_sq = tl.sum(k_i * k_i)
            alpha_i = (1.0 - tl.exp(-beta_val * k_norm_sq)) / (k_norm_sq + 1e-8)

            k_col = k_i[:, None]
            corr_w = tl.sum(k_col * S_KW, axis=0)
            corr_u = tl.sum(k_col * S_KU, axis=0)

            w_i = alpha_i * (k_i - corr_w)
            u_i = alpha_i * (v_i - corr_u)

            S_KW += k_col * w_i[None, :]
            S_KU += k_col * u_i[None, :]

    # Store A_trans = I - S_KW and KtU = S_KU
    si2 = tl.arange(0, S)[:, None]
    sj2 = tl.arange(0, S)[None, :]
    a_base = chunk_id * S * S + si2 * S + sj2

    A_val = tl.where(si2 == sj2, 1.0 - S_KW, -S_KW)
    tl.store(A_trans_ptr + a_base, A_val.to(A_trans_ptr.dtype.element_ty))
    tl.store(KtU_ptr + a_base, S_KU.to(KtU_ptr.dtype.element_ty))


@triton.jit
def recurrence_v9(
    Q_ptr,          # [B, T, H, S] — original layout
    A_trans_ptr,    # [BH*NT, S, S]
    KtU_ptr,        # [BH*NT, S, S]
    O_ptr,          # [B, T, H, S] — output in original layout
    h0_ptr,
    ht_ptr,
    T,
    NT,
    BT: tl.constexpr,
    S: tl.constexpr,
    BV: tl.constexpr,
    H: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    si = tl.arange(0, S)[:, None]
    vj = tl.arange(0, BV)[None, :]
    v_off = i_v * BV
    bt = tl.arange(0, BT)

    stride_b = T * H * S
    stride_t = H * S

    if USE_INITIAL_STATE:
        state = tl.load(h0_ptr + bh * S * S + si * S + (v_off + vj)).to(tl.float32)
    else:
        state = tl.zeros((S, BV), dtype=tl.float32)

    for c in range(NT):
        # Load A_trans [S, S]
        a_base = (bh * NT + c) * S * S
        a_ptr = a_base + si * S + tl.arange(0, S)[None, :]
        A = tl.load(A_trans_ptr + a_ptr).to(tl.float32)

        # Load KtU tile [S, BV]
        ktu_ptr = a_base + si * S + (v_off + vj)
        B = tl.load(KtU_ptr + ktu_ptr).to(tl.float32)

        # State update
        state = B + tl.dot(A, state)

        # Load Q from [B, T, H, S] layout — need BT rows
        t_start = c * BT
        q_base = b * stride_b + t_start * stride_t + h * S

        # Boundary mask for last chunk
        mask = (t_start + bt) < T
        q_ptr = q_base + bt[:, None] * stride_t + tl.arange(0, S)[None, :]
        q_i = tl.load(Q_ptr + q_ptr, mask=mask[:, None], other=0.0).to(tl.float32)

        # Output
        o = tl.dot(q_i, state)

        # Store O to [B, T, H, S] — only BV columns at offset v_off
        o_ptr = q_base + bt[:, None] * stride_t + (v_off + vj)
        tl.store(O_ptr + o_ptr, o.to(O_ptr.dtype.element_ty), mask=mask[:, None])

    if STORE_FINAL_STATE:
        tl.store(ht_ptr + bh * S * S + si * S + (v_off + vj),
                 state.to(ht_ptr.dtype.element_ty))


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
    chunk_size: int = 256,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del kwargs

    # Ensure bf16 to avoid shared memory overflow with fp32 (autocast)
    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        beta = beta.to(torch.bfloat16)

    B, T, H, S = q.shape
    BT = int(chunk_size)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BH = B * H
    n_chunks = BH * NT

    # Pre-allocate all outputs
    A_trans = torch.empty(n_chunks, S, S, device=q.device, dtype=q.dtype)
    KtU = torch.empty(n_chunks, S, S, device=q.device, dtype=q.dtype)
    o = torch.empty_like(q)

    h0 = None if initial_state is None else initial_state.reshape(BH, S, S)
    ht = torch.empty(BH, S, S, dtype=q.dtype, device=q.device) if output_final_state else None

    BV = 8
    grid_rec = (triton.cdiv(S, BV), BH)

    # Dynamic stages: S=64 fits stages=3, larger S needs fewer to avoid shared memory overflow
    rec_stages = 3 if S <= 64 else (2 if S <= 96 else 1)

    # Two-stream overlap: launch intra on stream s1, then recurrence on s2
    # s2 waits for s1 via wait_stream, hiding Python/launch overhead
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    # Launch intra-chunk on stream s1
    with torch.cuda.stream(s1):
        intra_chunk_v9[(n_chunks,)](
            k, v, beta,
            A_trans, KtU,
            T, NT=NT, BT=BT, S=S, H=H,
            num_warps=4,
            num_stages=1,
        )

    # Recurrence waits for intra to finish, then runs on s2
    s2.wait_stream(s1)
    with torch.cuda.stream(s2):
        recurrence_v9[grid_rec](
            q,
            A_trans, KtU,
            o,
            h0 if h0 is not None else q.new_empty(1),
            ht if ht is not None else q.new_empty(1),
            T, NT,
            BT=BT, S=S, BV=BV, H=H,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            num_warps=4,
            num_stages=rec_stages,
        )

    # Main stream waits for recurrence to complete
    torch.cuda.current_stream().wait_stream(s2)

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
