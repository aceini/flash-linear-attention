# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for QuasarAttention

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.utils import IS_AMD, autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, input_guard

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]
BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]


def naive_quasar_gate(
    beta: torch.Tensor,
    lambda_t: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Torch reference implementation for QuasarAttention gate computation.

    Computes: alpha = (1 - exp(-beta * lambda)) / (lambda + eps)

    Args:
        beta (torch.Tensor):
            Parameter tensor with `H` elements.
        lambda_t (torch.Tensor):
            Input tensor of shape `[..., H, 1]` (norm squared of keys).
        output_dtype (torch.dtype):
            Output dtype.

    Returns:
        Output tensor of shape `[..., H, 1]`.
    """
    eps = 1e-8
    alpha = (1 - torch.exp(-beta.view(-1, 1) * lambda_t)) / (lambda_t + eps)
    return alpha.to(output_dtype)


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in BT_LIST_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit
def quasar_gate_fwd_kernel(
    lambda_t,
    beta,
    alpha,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_beta = tl.load(beta + i_h).to(tl.float32)

    p_lambda = tl.make_block_ptr(lambda_t + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_alpha = tl.make_block_ptr(alpha + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    # [BT, BD]
    b_lambda = tl.load(p_lambda, boundary_check=(0, 1)).to(tl.float32)
    
    # alpha = (1 - exp(-beta * lambda)) / (lambda + eps)
    eps = 1e-8
    b_alpha = (1 - tl.exp(-b_beta * b_lambda)) / (b_lambda + eps)
    tl.store(p_alpha, b_alpha.to(p_alpha.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def quasar_gate_fwd(
    lambda_t: torch.Tensor,
    beta: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, K = lambda_t.shape[-2:]
    T = lambda_t.numel() // (H * K)

    alpha = torch.empty_like(lambda_t, dtype=output_dtype)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]), H)

    quasar_gate_fwd_kernel[grid](
        lambda_t=lambda_t,
        beta=beta,
        alpha=alpha,
        T=T,
        H=H,
        D=K,
        BD=triton.next_power_of_2(K),
    )
    return alpha


class QuasarGateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        lambda_t: torch.Tensor,
        beta: torch.Tensor,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        alpha = quasar_gate_fwd(
            lambda_t=lambda_t,
            beta=beta,
            output_dtype=output_dtype
        )
        ctx.save_for_backward(lambda_t, beta)
        return alpha

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dalpha: torch.Tensor):
        lambda_t, beta = ctx.saved_tensors
        eps = 1e-8
        
        # dalpha/dlambda and dalpha/dbeta derivatives
        beta_exp = torch.exp(-beta.view(-1, 1) * lambda_t)
        lambda_plus_eps = lambda_t + eps
        
        # dalpha/dlambda
        dlambda = (beta.view(-1, 1) * beta_exp * lambda_plus_eps - (1 - beta_exp)) / (lambda_plus_eps ** 2)
        
        # dalpha/dbeta
        dbeta = -lambda_t * beta_exp / lambda_plus_eps
        
        dlambda = dlambda * dalpha
        dbeta = dbeta.sum(dim=-2).sum(dim=-2)
        
        return dlambda, dbeta, None, None


@torch.compiler.disable
def fused_quasar_gate(
    lambda_t: torch.Tensor,
    beta: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Fused QuasarAttention gate computation with autograd support.

    Computes: alpha = (1 - exp(-beta * lambda)) / (lambda + eps)

    Args:
        lambda_t (torch.Tensor):
            Input tensor of shape `[..., H, 1]` (norm squared of keys).
        beta (torch.Tensor):
            Parameter tensor with `H` elements.

    Returns:
        Output tensor of shape `[..., H, 1]`.
    """
    return QuasarGateFunction.apply(lambda_t, beta, output_dtype)
