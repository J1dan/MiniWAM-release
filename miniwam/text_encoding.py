"""Text-encoder loading shared by extraction and deployment runtimes."""

from __future__ import annotations

from pathlib import Path

import torch

from miniwam.models.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from miniwam.prompting import encode_instruction_features


def _load_state_dict(path: str | Path, device: str | torch.device = "cpu"):
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device=str(device))
    return torch.load(path, map_location=device, weights_only=True)


def load_text_encoder(
    path: str | Path,
    tokenizer_id: str,
    context_len: int,
    device: str | torch.device,
    dtype: torch.dtype,
):
    """Load the frozen UMT5 encoder/tokenizer used by MiniWAM."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Text encoder checkpoint does not exist: {path}")
    model = WanTextEncoder()
    model.load_state_dict(_load_state_dict(path), strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_id,
        seq_len=int(context_len),
        clean="whitespace",
    )
    return model, tokenizer


def encode_text(text_encoder, tokenizer, instruction, device, dtype):
    """Encode one instruction with the canonical MiniWAM prompt contract."""
    return encode_instruction_features(
        text_encoder, tokenizer, instruction, device, dtype,
    )
