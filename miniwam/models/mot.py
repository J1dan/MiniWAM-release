"""
mot.py

Mixture-of-Transformers (MoT) for MiniWAM.
Adapted from FastWAM: fastwam/models/wan22/mot.py

The MoT takes multiple expert token streams, concatenates their Q/K/V for shared
self-attention (with a custom attention mask), then routes the output slices back
to each expert's post-attention FFN. This enables cross-expert information flow
through attention while keeping FFN/modulation expert-specific.

Constraint: all experts must share `num_heads` and `attn_head_dim`.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .dit_block import flash_attention, modulate, rope_apply

logger = logging.getLogger(__name__)


class MoT(nn.Module):
    """Mixture-of-Transformers with arbitrary expert names."""

    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")

        # Use a plain dictionary instead of nn.ModuleDict to avoid duplicate registration of
        # mixtures (world_dit, action_dit) in PyTorch's module hierarchy, which causes
        # recursive wrapping assertions in FSDP to fail.
        self.mixtures = mixtures
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; "
                    f"'{name}' has {len(expert.blocks)}, expected {self.num_layers}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; "
                    f"'{name}' has {expert.num_heads}, expected {self.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    f"All experts must have same attn_head_dim; "
                    f"'{name}' has {expert.attn_head_dim}, expected {self.attn_head_dim}"
                )

        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            n_params = sum(p.numel() for p in expert.parameters())
            logger.info(f"  Expert '{name}': {n_params / 1e6:.1f}M params")

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            base_mod + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            shift_mlp = shift_mlp.squeeze(2)
            scale_mlp = scale_mlp.squeeze(2)
            gate_mlp = gate_mlp.squeeze(2)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q, k, v):
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward, q_cat, k_cat, v_cat, use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(self, expert, block, x, freqs, t_mod):
        """Build per-expert Q/K/V + cache modulation state for post-block."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(block, t_mod)
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_ckpt = bool(getattr(expert, "use_gradient_checkpointing", False))
        return q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_ckpt

    def _apply_post_with_optional_checkpoint(
        self, block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp,
        use_gradient_checkpointing, mixed_slice, context_payload,
    ) -> torch.Tensor:
        def _post_fn(_mixed, _x, _g_msa, _s_mlp, _sc_mlp, _g_mlp):
            return self._apply_expert_post_block(
                block=block, residual_x=_x, mixed_attn_out=_mixed,
                gate_msa=_g_msa, shift_mlp=_s_mlp, scale_mlp=_sc_mlp,
                gate_mlp=_g_mlp, context_payload=context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn, mixed_slice, residual_x, gate_msa,
                shift_mlp, scale_mlp, gate_mlp, use_reentrant=False,
            )
        return _post_fn(mixed_slice, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Joint forward pass across all experts.

        Args:
            embeds_all:  {expert_name: [B, S_i, D_i]} token embeddings per expert
            attention_mask: [total_S, total_S] boolean mask for mixed attention
            freqs_all:   {expert_name: RoPE freqs} per expert
            context_all: {expert_name: {"context": ..., "mask": ...}} for cross-attn
            t_mod_all:   {expert_name: t_mod tensor} time modulation per expert

        Returns:
            {expert_name: [B, S_i, D_i]} updated tokens per expert
        """
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")

        if attention_mask.ndim == 2:
            mask_seq = attention_mask.shape[0]
            is_square = attention_mask.shape[0] == attention_mask.shape[1]
        elif attention_mask.ndim == 4:
            mask_seq = attention_mask.shape[-1]
            is_square = attention_mask.shape[-1] == attention_mask.shape[-2]
        else:
            mask_seq = -1
            is_square = False
        if not is_square:
            raise ValueError(
                f"`attention_mask` must be 2D square or [B,1,S,S], got shape {tuple(attention_mask.shape)}"
            )

        tokens_all = dict(embeds_all)

        for layer_idx in range(self.num_layers):
            q_chunks, k_chunks, v_chunks = [], [], []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]

                q, k, v, res_x, g_msa, s_mlp, sc_mlp, g_mlp, use_ckpt = (
                    self._build_expert_attention_io(
                        expert, block, x, freqs_all[name], t_mod_all[name]
                    )
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": res_x,
                    "gate_msa": g_msa,
                    "shift_mlp": s_mlp,
                    "scale_mlp": sc_mlp,
                    "gate_mlp": g_mlp,
                    "use_ckpt": use_ckpt,
                }

            # Concatenate all expert tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if mask_seq != total_seq:
                raise ValueError(
                    f"Attention mask seq length mismatch: "
                    f"mask={mask_seq} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attention_mask)

            # Split output back to each expert and apply post-block
            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len
                c = cached[name]
                tokens_all[name] = self._apply_post_with_optional_checkpoint(
                    block=c["block"],
                    residual_x=c["residual_x"],
                    gate_msa=c["gate_msa"],
                    shift_mlp=c["shift_mlp"],
                    scale_mlp=c["scale_mlp"],
                    gate_mlp=c["gate_mlp"],
                    use_gradient_checkpointing=c["use_ckpt"],
                    mixed_slice=mixed[:, start:end, :],
                    context_payload=context_all.get(name),
                )
                start = end

        return tokens_all
