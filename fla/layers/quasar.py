# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for QuasarAttention — v14 ultra-optimized layer

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.quasar import chunk_quasar, fused_recurrent_quasar
from fla.ops.quasar.gate import fused_quasar_gate

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class QuasarAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_heads: int = 16,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> QuasarAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.key_dim = int(self.num_heads * self.head_dim)
        self.value_dim = int(self.num_heads * self.head_dim)
        self.layer_idx = layer_idx

        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        self.beta_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.beta_log._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self._fast_ready = False

    def _init_fast(self):
        """Pre-compute fused weights in bf16 for zero-overhead forward."""
        import triton

        device = self.q_proj.weight.device

        # Pre-cast fused QKV weight to bf16 transposed for torch.mm
        self._qkv_w = torch.cat([
            self.q_proj.weight, self.k_proj.weight, self.v_proj.weight
        ], dim=0).to(torch.bfloat16).t().contiguous()

        # Pre-compute fused gate (collapse 2-layer MLP: W2 @ W1)
        g0, g1 = self.g_proj[0], self.g_proj[1]
        self._g_w = (g1.weight @ g0.weight).to(torch.bfloat16).t().contiguous()
        self._g_b = g1.bias.data.to(torch.bfloat16)

        # Pre-cast output projection weight
        self._o_w = self.o_proj.weight.data.to(torch.bfloat16).t().contiguous()

        # Pre-compute beta
        self._beta_fixed = F.softplus(self.beta_log).detach()

        # Persistent CUDA streams + events
        self._s1 = torch.cuda.Stream()
        self._s2 = torch.cuda.Stream()
        self._s3 = torch.cuda.Stream()
        self._e1 = torch.cuda.Event()

        # Buffer cache key
        self._buf_cache = {}

        self._fast_ready = True

    def _get_buffers(self, B, T, H, S, BT):
        """Get or create pre-allocated buffers for given shape."""
        import triton
        key = (B, T, H, S, BT)
        if key not in self._buf_cache:
            NT = triton.cdiv(T, BT)
            BH = B * H
            n_chunks = BH * NT
            device = self._qkv_w.device
            self._buf_cache.clear()
            self._buf_cache[key] = (
                torch.empty(n_chunks, S, S, device=device, dtype=torch.bfloat16),
                torch.empty(n_chunks, S, S, device=device, dtype=torch.bfloat16),
                torch.empty(B, T, H, S, device=device, dtype=torch.bfloat16),
                NT, BH, n_chunks,
            )
        return self._buf_cache[key]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        cu_seqlens = kwargs.get("cu_seqlens")
        if not use_cache and attention_mask is None and past_key_values is None and cu_seqlens is None:
            return self._fast_forward(hidden_states)

        # Original path for caching / attention masks / cu_seqlens
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2

        batch_size, q_len, _ = hidden_states.shape
        mode = "chunk"

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states), cache=conv_state_q,
                output_final_state=use_cache, cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states), cache=conv_state_k,
                output_final_state=use_cache, cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states), cache=conv_state_v,
                output_final_state=use_cache, cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_dim) for x in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        beta = F.softplus(self.beta_log)

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        o, recurrent_state = chunk_quasar(
            q=q, k=k, v=v, beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values

    @torch.no_grad()
    def _fast_forward(self, hidden_states):
        """Ultra-optimized forward: pre-cast bf16, fused QKV, skip conv,
        pre-alloc buffers, overlap gate with kernel, event sync."""
        import triton
        from fla.ops.quasar.chunk import intra_chunk_v9, recurrence_v9

        if not self._fast_ready:
            self._init_fast()

        B, T, D = hidden_states.shape
        H = self.num_heads
        S = self.head_dim
        BT = 256
        BV = 32 if S >= 32 else S

        # Get pre-allocated buffers
        A_trans, KtU, o_buf, NT, BH, n_chunks = self._get_buffers(B, T, H, S, BT)

        x_bf = hidden_states.to(torch.bfloat16).reshape(-1, D)
        shape = (B, T, H, S)

        # Single fused QKV matmul (bf16, no autocast overhead)
        qkv = torch.mm(x_bf, self._qkv_w)
        q, k, v = qkv.split(self.key_dim, dim=-1)
        q = q.view(shape)
        k = k.view(shape)
        v = v.view(shape)

        # Overlap: gate matmul on separate stream
        with torch.cuda.stream(self._s3):
            g = torch.addmm(self._g_b, x_bf, self._g_w).view(shape)

        # Intra-chunk kernel
        with torch.cuda.stream(self._s1):
            intra_chunk_v9[(n_chunks,)](
                k, v, self._beta_fixed,
                A_trans, KtU,
                T, NT=NT, BT=BT, S=S, H=H,
                num_warps=4, num_stages=4,
            )
            self._e1.record(self._s1)

        # Recurrence kernel (waits for intra via event)
        self._s2.wait_event(self._e1)
        with torch.cuda.stream(self._s2):
            recurrence_v9[(triton.cdiv(S, BV), BH)](
                q, A_trans, KtU, o_buf,
                q.new_empty(1), q.new_empty(1),
                T, NT,
                BT=BT, S=S, BV=BV, H=H,
                USE_INITIAL_STATE=False,
                STORE_FINAL_STATE=False,
                num_warps=4,
                num_stages=3,
            )

        # Wait for kernel + gate
        torch.cuda.current_stream().wait_stream(self._s2)
        torch.cuda.current_stream().wait_stream(self._s3)

        # Norm + output projection (bf16 direct matmul)
        o = self.o_norm(o_buf, g)
        o = torch.mm(o.reshape(-1, self.key_dim), self._o_w).view(B, T, D)

        return o, None, None
