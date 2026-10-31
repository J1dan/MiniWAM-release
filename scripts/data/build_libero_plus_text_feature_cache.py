#!/usr/bin/env python3
"""Build an exact packed BF16 instruction cache for LIBERO-Plus tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from miniwam.libero_plus_text_cache import (  # noqa: E402
    LIBERO_PLUS_INSTRUCTION_CONTRACT,
    VANILLA_CACHE_COMPARISON,
    build_packed_payload,
    compare_to_vanilla_cache,
    registry_manifest_id,
    save_packed_payload,
)
from miniwam.prompting import (  # noqa: E402
    encode_instruction_features,
    load_cached_instruction_features,
)
from scripts.data.extraction_utils import load_text_encoder, make_text_provenance  # noqa: E402
from scripts.eval.libero_plus_protocol import (  # noqa: E402
    CLASSIFICATION_SHA256,
    LIBERO_PLUS_REV,
    VANILLA_LIBERO_REV,
    configure_libero_plus,
    instantiate_suite,
    load_registry,
    load_task_manifest,
    load_vanilla_instruction_registry,
    resolve_libero_plus_instructions,
    sha256_file,
    validate_manifest_against_registry,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero_plus_root", type=Path, default=PROJECT_ROOT / "LIBERO-plus")
    parser.add_argument("--libero_root", type=Path, default=PROJECT_ROOT / "LIBERO")
    parser.add_argument("--task_manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--text_encoder_path", type=Path,
        default=PROJECT_ROOT / "pretrained/language/models_t5_umt5-xxl-enc-bf16.pth",
    )
    parser.add_argument(
        "--tokenizer_id", default="google/umt5-xxl",
        help="Pinned tokenizer ID or a local snapshot directory.",
    )
    parser.add_argument(
        "--tokenizer_revision", default="66cb9e7e85526fe440a945569e42c72fb6cbc0ad",
    )
    parser.add_argument(
        "--baseline_text_feature_cache",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/libero/libero_instruction_features.pt",
        help="MiniWAM LIBERO cache that unperturbed instructions are checked against.",
    )
    parser.add_argument("--context_len", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--config_dir", type=Path, default=PROJECT_ROOT / ".libero_plus")
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.absolute().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def main():
    args = parse_args()
    registry = load_registry(args.libero_plus_root)
    vanilla_instructions = load_vanilla_instruction_registry(args.libero_root)
    baseline_cache_path = args.baseline_text_feature_cache.expanduser().resolve()
    if not baseline_cache_path.is_file():
        raise FileNotFoundError(
            f"Existing MiniWAM vanilla text cache does not exist: {baseline_cache_path}."
        )
    if args.task_manifest is None:
        records = registry
        manifest_id = registry_manifest_id(records)
    else:
        records, metadata = load_task_manifest(args.task_manifest)
        validate_manifest_against_registry(records, registry)
        manifest_id = metadata["manifest_id"]

    benchmark = configure_libero_plus(
        args.libero_plus_root, config_dir=args.config_dir, require_assets=False,
    )
    suites = {
        suite: instantiate_suite(benchmark, suite)
        for suite in {record.suite for record in records}
    }
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    tokenizer_path = Path(args.tokenizer_id).expanduser()
    if tokenizer_path.is_dir():
        resolved_tokenizer = str(tokenizer_path)
    else:
        resolved_tokenizer = snapshot_download(
            repo_id=args.tokenizer_id,
            revision=args.tokenizer_revision,
            allow_patterns=("tokenizer.json", "tokenizer_config.json", "spiece.model", "special_tokens_map.json"),
        )
    text_encoder, tokenizer = load_text_encoder(
        args.text_encoder_path, resolved_tokenizer, args.context_len, device, dtype,
    )
    provenance = make_text_provenance(
        args.text_encoder_path, args.tokenizer_id, args.context_len,
    )
    provenance["tokenizer_revision"] = args.tokenizer_revision
    provenance["instruction_contract"] = LIBERO_PLUS_INSTRUCTION_CONTRACT
    provenance["vanilla_libero_revision"] = VANILLA_LIBERO_REV
    provenance["vanilla_cache_path"] = _display_path(args.baseline_text_feature_cache.expanduser())
    provenance["vanilla_cache_sha256"] = sha256_file(baseline_cache_path)

    feature_rows = []
    encoded = {}
    vanilla_pairs = {}
    for record in tqdm(records, desc="LIBERO-Plus scalar text encoding"):
        task = suites[record.suite].get_task(record.task_id)
        instructions = resolve_libero_plus_instructions(
            record, task, vanilla_instructions,
        )
        cached = encoded.get(instructions.clean)
        if cached is None:
            cached = encode_instruction_features(
                text_encoder, tokenizer, instructions.clean, device, dtype,
            )
            encoded[instructions.clean] = cached
        if instructions.source == "vanilla_libero_task_language":
            baseline_key = (record.suite, instructions.clean)
            if baseline_key not in vanilla_pairs:
                baseline_context, baseline_mask = load_cached_instruction_features(
                    baseline_cache_path,
                    f"{record.suite}_no_noops_lerobot",
                    instructions.clean,
                    args.context_len,
                    int(cached[0].shape[1]),
                )
                vanilla_pairs[baseline_key] = (
                    f"{record.suite}: {instructions.clean!r}",
                    cached[0], cached[1], baseline_context, baseline_mask,
                )
        feature_rows.append((record, instructions, cached[0], cached[1]))

    comparison = compare_to_vanilla_cache(vanilla_pairs.values())
    provenance["vanilla_cache_comparison"] = VANILLA_CACHE_COMPARISON
    provenance["vanilla_cache_comparison_stats"] = comparison
    provenance["vanilla_cache_verified"] = True

    payload = build_packed_payload(
        task_features=feature_rows,
        benchmark_revision=LIBERO_PLUS_REV,
        classification_sha256=CLASSIFICATION_SHA256,
        manifest_id=manifest_id,
        text_provenance=provenance,
    )
    save_packed_payload(payload, args.output)
    print(
        f"Saved {len(records)} tasks / {len(encoded)} model-facing instructions to {args.output}; "
        f"{comparison['instructions']} unperturbed instructions match the LIBERO cache "
        f"({comparison['bitwise_equal']} bitwise, mean relative error "
        f"{comparison['mean_relative_error']:.4f}) and all packed BF16 round trips are exact."
    )


if __name__ == "__main__":
    main()
