"""Packed, baseline-equivalent instruction features for LIBERO-Plus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from miniwam.prompting import DEFAULT_PROMPT, format_instruction_prompt


LIBERO_PLUS_TEXT_CACHE_SCHEMA_VERSION = 2
LIBERO_PLUS_TEXT_CACHE_FORMAT = "packed_active_tokens_bf16"
LIBERO_PLUS_INSTRUCTION_CONTRACT = "vanilla_libero_style_except_language_perturbations_v1"
VANILLA_LIBERO_REV = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
VANILLA_CACHE_MAX_MEAN_RELATIVE_ERROR = 0.1
VANILLA_CACHE_COMPARISON = "masks_equal_and_mean_relative_error_le_0.1"


def compare_to_vanilla_cache(pairs) -> dict:
    """Check fresh encodings of unperturbed instructions against the LIBERO cache.

    The LIBERO cache in the released checkpoint was written during feature
    extraction, and a fresh encoding can differ from it slightly, so bitwise
    equality cannot be required. Masks must match exactly and the mean relative
    error of the contexts must be at most VANILLA_CACHE_MAX_MEAN_RELATIVE_ERROR.
    BF16 encodings on another GPU type or the CPU differ by about 0.05, while
    two different instructions of the same length differ by 0.28 or more.
    A wrong prompt template, tokenizer or encoder changes every embedding and
    fails this check.
    """
    errors = []
    bitwise_equal = 0
    for label, context, mask, baseline_context, baseline_mask in pairs:
        if not torch.equal(mask, baseline_mask):
            raise ValueError(
                f"Token mask differs from the MiniWAM vanilla LIBERO text cache for {label}; "
                "check the prompt template and tokenizer."
            )
        active, reference = context[mask].float(), baseline_context[baseline_mask].float()
        errors.append(((active - reference).norm() / reference.norm()).item())
        bitwise_equal += bool(torch.equal(context, baseline_context))
    if not errors:
        raise ValueError("No unperturbed instruction to compare with the vanilla LIBERO text cache.")
    mean_error = sum(errors) / len(errors)
    if mean_error > VANILLA_CACHE_MAX_MEAN_RELATIVE_ERROR:
        raise ValueError(
            f"Unperturbed instructions differ from the MiniWAM vanilla LIBERO text cache "
            f"(mean relative error {mean_error:.4f} > {VANILLA_CACHE_MAX_MEAN_RELATIVE_ERROR}); "
            "check the prompt template, tokenizer and text encoder."
        )
    return {
        "instructions": len(errors),
        "bitwise_equal": bitwise_equal,
        "mean_relative_error": mean_error,
        "max_relative_error": max(errors),
    }


def registry_manifest_id(records) -> str:
    payload = [
        {
            "suite": record.suite,
            "task_id": record.task_id,
            "official_id": record.official_id,
            "task_name": record.task_name,
            "category": record.category,
            "difficulty": record.difficulty,
        }
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_packed_payload(
    *,
    task_features,
    benchmark_revision: str,
    classification_sha256: str,
    manifest_id: str,
    text_provenance: dict,
) -> dict:
    """Pack canonical BF16 contexts and prove the pack/unpack round trip is exact."""
    if text_provenance.get("vanilla_cache_verified") is not True:
        raise ValueError(
            "LIBERO-Plus cache requires a check against the existing "
            "MiniWAM vanilla LIBERO text cache."
        )
    task_features = list(task_features)
    entries = {}
    packed_parts = []
    encoded_by_instruction = {}
    offset = 0
    context_len = text_dim = None
    for record, instructions, context, mask in task_features:
        if context.dtype != torch.bfloat16 or mask.dtype != torch.bool:
            raise TypeError("LIBERO-Plus cache input must use canonical BF16 context/bool mask.")
        if context.ndim != 2 or mask.shape != (context.shape[0],):
            raise ValueError(f"Invalid text feature shape for {record.key}: {context.shape}, {mask.shape}.")
        if context_len is None:
            context_len, text_dim = context.shape
        if context.shape != (context_len, text_dim):
            raise ValueError(f"Inconsistent text feature shape for {record.key}: {context.shape}.")
        if torch.count_nonzero(context[~mask]).item() != 0:
            raise ValueError(f"Canonical text context has nonzero padding for {record.key}.")

        cached = encoded_by_instruction.get(instructions.clean)
        if cached is None:
            active = context[mask].contiguous()
            start, length = offset, int(active.shape[0])
            packed_parts.append(active)
            offset += length
            encoded_by_instruction[instructions.clean] = (
                start, length, context.clone(), mask.clone(),
            )
        else:
            start, length, existing_context, existing_mask = cached
            if not torch.equal(context, existing_context) or not torch.equal(mask, existing_mask):
                raise ValueError(
                    f"Duplicate instruction encoded differently: {instructions.clean!r}."
                )
        entries[record.key] = {
            "suite": record.suite,
            "task_id": int(record.task_id),
            "official_id": int(record.official_id),
            "task_name": record.task_name,
            "raw_instruction": instructions.raw,
            "clean_instruction": instructions.clean,
            "base_instruction": instructions.base,
            "instruction_source": instructions.source,
            "formatted_prompt": format_instruction_prompt(instructions.clean),
            "start": int(start),
            "length": int(length),
        }

    if not entries:
        raise ValueError("Cannot build an empty LIBERO-Plus text cache.")
    packed = torch.cat(packed_parts, dim=0).contiguous()
    payload = {
        "schema_version": LIBERO_PLUS_TEXT_CACHE_SCHEMA_VERSION,
        "format": LIBERO_PLUS_TEXT_CACHE_FORMAT,
        "benchmark_revision": benchmark_revision,
        "classification_sha256": classification_sha256,
        "manifest_id": manifest_id,
        "prompt_template": DEFAULT_PROMPT,
        "context_len": int(context_len),
        "text_dim": int(text_dim),
        "storage_dtype": "bfloat16",
        "instruction_contract": text_provenance.get("instruction_contract"),
        "vanilla_libero_revision": text_provenance.get("vanilla_libero_revision"),
        "text_provenance": dict(text_provenance),
        "equivalence": {
            "reference": "miniwam.prompting.encode_instruction_features",
            "comparison": "torch.equal_after_canonical_bf16_cast",
            "verified": True,
            "vanilla_cache_comparison": text_provenance.get(
                "vanilla_cache_comparison", "torch.equal_context_and_mask",
            ),
            "vanilla_cache_verified": True,
        },
        "entries": entries,
        "packed_context": packed,
    }

    cache = LiberoPlusTextFeatureCache(payload=payload, source="in-memory payload")
    for record, instructions, context, mask in task_features:
        restored_context, restored_mask = cache.get(
            record,
            instructions,
            expected_context_len=context_len,
            expected_text_dim=text_dim,
        )
        if not torch.equal(restored_context, context) or not torch.equal(restored_mask, mask):
            raise AssertionError(f"Packed BF16 cache is not baseline-equivalent for {record.key}.")
    return payload


class LiberoPlusTextFeatureCache:
    def __init__(self, path: Path | None = None, *, payload=None, source=None):
        if payload is None:
            path = Path(path).expanduser()
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            source = str(path)
        self.payload = payload
        self.source = source
        self._validate_header()

    def _validate_header(self):
        payload = self.payload
        if payload.get("schema_version") != LIBERO_PLUS_TEXT_CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported LIBERO-Plus text cache schema in {self.source}.")
        if payload.get("format") != LIBERO_PLUS_TEXT_CACHE_FORMAT:
            raise ValueError(f"Unsupported LIBERO-Plus text cache format in {self.source}.")
        if payload.get("prompt_template") != DEFAULT_PROMPT:
            raise ValueError(f"Prompt template mismatch in {self.source}.")
        if payload.get("storage_dtype") != "bfloat16":
            raise ValueError(f"LIBERO-Plus cache is not canonical BF16 in {self.source}.")
        if payload.get("instruction_contract") != LIBERO_PLUS_INSTRUCTION_CONTRACT:
            raise ValueError(f"LIBERO-Plus cache instruction contract mismatch in {self.source}.")
        if payload.get("vanilla_libero_revision") != VANILLA_LIBERO_REV:
            raise ValueError(f"LIBERO-Plus cache vanilla LIBERO revision mismatch in {self.source}.")
        equivalence = payload.get("equivalence", {})
        if equivalence.get("verified") is not True:
            raise ValueError(f"LIBERO-Plus cache lacks numerical-equivalence certification: {self.source}.")
        if equivalence.get("vanilla_cache_verified") is not True:
            raise ValueError(
                f"LIBERO-Plus cache lacks existing-MiniWAM-cache equivalence: {self.source}."
            )
        packed = payload.get("packed_context")
        if not isinstance(packed, torch.Tensor) or packed.ndim != 2 or packed.dtype != torch.bfloat16:
            raise TypeError(f"Invalid packed context tensor in {self.source}.")
        if packed.shape[1] != int(payload.get("text_dim", -1)):
            raise ValueError(f"Packed text dimension mismatch in {self.source}.")
        if not isinstance(payload.get("entries"), dict):
            raise TypeError(f"Invalid task index in {self.source}.")

    @property
    def manifest_id(self) -> str:
        return str(self.payload["manifest_id"])

    def assert_protocol(
        self,
        *,
        benchmark_revision: str,
        classification_sha256: str,
        manifest_id: str | None = None,
    ) -> None:
        expected = {
            "benchmark_revision": benchmark_revision,
            "classification_sha256": classification_sha256,
        }
        if manifest_id is not None:
            expected["manifest_id"] = manifest_id
        for key, value in expected.items():
            if self.payload.get(key) != value:
                raise ValueError(
                    f"LIBERO-Plus text cache {key}={self.payload.get(key)!r}, expected {value!r}."
                )

    def get(self, record, instructions, *, expected_context_len: int, expected_text_dim: int):
        entry = self.payload["entries"].get(record.key)
        if not isinstance(entry, dict):
            raise KeyError(f"Text cache {self.source} has no task {record.key}.")
        exact = {
            "suite": record.suite,
            "task_id": int(record.task_id),
            "official_id": int(record.official_id),
            "task_name": record.task_name,
            "raw_instruction": instructions.raw,
            "clean_instruction": instructions.clean,
            "base_instruction": instructions.base,
            "instruction_source": instructions.source,
            "formatted_prompt": format_instruction_prompt(instructions.clean),
        }
        for key, value in exact.items():
            if entry.get(key) != value:
                raise ValueError(
                    f"Text cache exact {key} mismatch for {record.key}: "
                    f"{entry.get(key)!r} != {value!r}."
                )
        context_len = int(self.payload["context_len"])
        text_dim = int(self.payload["text_dim"])
        if (context_len, text_dim) != (int(expected_context_len), int(expected_text_dim)):
            raise ValueError(
                f"Text cache shape contract {(context_len, text_dim)} != "
                f"{(expected_context_len, expected_text_dim)}."
            )
        start, length = int(entry["start"]), int(entry["length"])
        active = self.payload["packed_context"][start:start + length]
        if active.shape != (length, text_dim) or length > context_len:
            raise ValueError(f"Invalid packed span for {record.key} in {self.source}.")
        context = torch.zeros((context_len, text_dim), dtype=torch.bfloat16)
        mask = torch.zeros((context_len,), dtype=torch.bool)
        context[:length] = active
        mask[:length] = True
        return context.contiguous(), mask.contiguous()


def save_packed_payload(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
