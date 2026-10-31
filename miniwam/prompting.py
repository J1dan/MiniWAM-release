"""Shared instruction prompt formatting and cached text features for MiniWAM."""

from pathlib import Path
from typing import Optional

import torch

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
TEXT_FEATURE_CACHE_FILENAME = "libero_instruction_features.pt"
TEXT_FEATURE_CACHE_SCHEMA_VERSION = 1


def format_instruction_prompt(instruction: str) -> str:
    return DEFAULT_PROMPT.format(task=str(instruction).strip())


@torch.no_grad()
def encode_instruction_features(text_encoder, tokenizer, instruction, device, dtype):
    """Canonical MiniWAM scalar instruction encoder and BF16 storage boundary."""
    prompt = format_instruction_prompt(instruction)
    ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device=device, dtype=torch.bool)
    context = text_encoder(ids, mask)
    context[~mask.unsqueeze(-1).expand_as(context)] = 0.0
    return (
        context[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        mask[0].detach().to(device="cpu", dtype=torch.bool).contiguous(),
    )


def resolve_text_feature_cache(
    explicit_path: Optional[str],
    checkpoint_dir: Path,
) -> Optional[Path]:
    """Return an explicit cache, or the one stored in the checkpoint folder if present."""
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"LIBERO text feature cache does not exist: {path}")
        return path
    candidate = Path(checkpoint_dir) / TEXT_FEATURE_CACHE_FILENAME
    return candidate if candidate.is_file() else None


def load_cached_instruction_features(
    cache_path: Path,
    dataset_name: str,
    instruction: str,
    expected_context_len: int,
    expected_text_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one instruction embedding and enforce the extraction/eval contract."""
    cache_path = Path(cache_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema_version") != TEXT_FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported text feature cache schema in {cache_path}: "
            f"{payload.get('schema_version')!r}."
        )
    if payload.get("prompt_template") != DEFAULT_PROMPT:
        raise ValueError(f"Prompt template mismatch in text feature cache {cache_path}.")
    if int(payload.get("context_len", -1)) != int(expected_context_len):
        raise ValueError(
            f"Text cache context_len={payload.get('context_len')}, "
            f"expected {expected_context_len}."
        )
    if int(payload.get("text_dim", -1)) != int(expected_text_dim):
        raise ValueError(
            f"Text cache text_dim={payload.get('text_dim')}, expected {expected_text_dim}."
        )

    if not isinstance(instruction, str):
        raise TypeError(
            f"Instruction lookup requires an exact Python string, got {type(instruction).__name__}."
        )
    datasets = payload.get("datasets", {})
    entries = datasets.get(dataset_name)
    if not isinstance(entries, dict):
        raise KeyError(f"Text feature cache {cache_path} has no dataset {dataset_name!r}.")
    entry = entries.get(instruction)
    if not isinstance(entry, dict):
        raise KeyError(
            f"Text feature cache {cache_path} has no exact instruction match for {instruction!r}."
        )
    expected_prompt = format_instruction_prompt(instruction)
    if entry.get("formatted_prompt") != expected_prompt:
        raise ValueError(f"Formatted prompt mismatch for {instruction!r} in {cache_path}.")

    context, mask = entry.get("context"), entry.get("mask")
    if not isinstance(context, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError(f"Invalid text tensors for {instruction!r} in {cache_path}.")
    if context.shape != (expected_context_len, expected_text_dim):
        raise ValueError(f"Invalid cached context shape {tuple(context.shape)} for {instruction!r}.")
    if mask.shape != (expected_context_len,) or mask.dtype != torch.bool:
        raise ValueError(f"Invalid cached mask for {instruction!r}: {tuple(mask.shape)}, {mask.dtype}.")
    if not torch.isfinite(context).all():
        raise ValueError(f"Cached context for {instruction!r} contains non-finite values.")
    if torch.count_nonzero(context[~mask]).item() != 0:
        raise ValueError(f"Cached context for {instruction!r} has nonzero masked positions.")
    return context.contiguous(), mask.contiguous()
