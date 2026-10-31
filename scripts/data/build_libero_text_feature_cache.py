#!/usr/bin/env python
"""Build a compact, validated LIBERO instruction-embedding cache."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
LIBERO_ROOT = PROJECT_ROOT / "LIBERO"
if LIBERO_ROOT.exists():
    sys.path.insert(0, str(LIBERO_ROOT))

from miniwam.prompting import (
    DEFAULT_PROMPT,
    TEXT_FEATURE_CACHE_FILENAME,
    TEXT_FEATURE_CACHE_SCHEMA_VERSION,
    format_instruction_prompt,
)
from miniwam.instruction_contract import require_instruction_string


LIBERO_DATASETS = (
    "libero_10_no_noops_lerobot",
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features_root", type=Path, required=True,
        help="Parent containing the four per-suite LIBERO feature directories.",
    )
    parser.add_argument(
        "--output_path", type=Path, default=None,
        help=f"Output path (default: FEATURES_ROOT/{TEXT_FEATURE_CACHE_FILENAME}).",
    )
    parser.add_argument(
        "--dataset_names", default=",".join(LIBERO_DATASETS),
        help="Comma-separated extracted LIBERO dataset directory names.",
    )
    parser.add_argument(
        "--verify_all_duplicates", action=argparse.BooleanOptionalAction, default=True,
        help="Require every episode for an instruction to contain identical text tensors.",
    )
    parser.add_argument(
        "--verify_benchmark", action=argparse.BooleanOptionalAction, default=True,
        help="Require extracted instruction sets to equal the installed LIBERO benchmark.",
    )
    return parser.parse_args()


def _validate_text_payload(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if set(payload) != {"context", "mask"}:
        raise KeyError(f"Unexpected text payload keys in {path}: {sorted(payload)}")
    context, mask = payload["context"], payload["mask"]
    if context.ndim != 2 or mask.shape != (context.shape[0],) or mask.dtype != torch.bool:
        raise ValueError(
            f"Invalid text tensors in {path}: context={tuple(context.shape)}, "
            f"mask={tuple(mask.shape)} {mask.dtype}."
        )
    if not torch.isfinite(context).all():
        raise ValueError(f"Non-finite text embedding in {path}.")
    if torch.count_nonzero(context[~mask]).item() != 0:
        raise ValueError(f"Masked text positions are nonzero in {path}.")
    return context.contiguous(), mask.contiguous()


def _benchmark_instruction_sets(dataset_names):
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    result = {}
    for dataset_name in dataset_names:
        suite_name = dataset_name.removesuffix("_no_noops_lerobot")
        if suite_name not in benchmark_dict:
            raise KeyError(f"Installed LIBERO has no benchmark suite {suite_name!r}.")
        suite = benchmark_dict[suite_name]()
        result[dataset_name] = {
            require_instruction_string(
                suite.get_task(task_id).language,
                f"LIBERO benchmark {suite_name} task {task_id}",
            )
            for task_id in range(int(suite.n_tasks))
        }
    return result


def build_cache(
    features_root: Path,
    dataset_names,
    output_path: Path,
    verify_all_duplicates=True,
    verify_benchmark=True,
):
    features_root = Path(features_root)
    entries_by_dataset = {}
    provenance_by_dataset = {}
    all_declared_provenance = set()
    context_len = text_dim = None

    for dataset_name in dataset_names:
        suite_root = features_root / dataset_name
        metadata_paths = sorted(suite_root.glob("episode_*/metadata.json"))
        if not metadata_paths:
            raise FileNotFoundError(f"No extracted episodes found under {suite_root}.")
        entries = {}
        provenance_records = set()
        legacy_episodes = 0
        for metadata_path in tqdm(metadata_paths, desc=f"Text cache: {dataset_name}"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("dataset_name") != dataset_name:
                raise ValueError(
                    f"{metadata_path} has dataset_name={metadata.get('dataset_name')!r}, "
                    f"expected {dataset_name!r}."
                )
            instruction = require_instruction_string(
                metadata.get("language_instruction"), str(metadata_path),
            )
            text_path = metadata_path.parent / "text_embed.pt"
            context, mask = _validate_text_payload(text_path)
            current_context_len, current_text_dim = context.shape
            if context_len is None:
                context_len, text_dim = current_context_len, current_text_dim
            if (current_context_len, current_text_dim) != (context_len, text_dim):
                raise ValueError(
                    f"Inconsistent text shape in {text_path}: {tuple(context.shape)}, "
                    f"expected {(context_len, text_dim)}."
                )

            existing = entries.get(instruction)
            if existing is None:
                entries[instruction] = {
                    "formatted_prompt": format_instruction_prompt(instruction),
                    "context": context.clone(),
                    "mask": mask.clone(),
                }
            elif verify_all_duplicates and (
                not torch.equal(existing["context"], context)
                or not torch.equal(existing["mask"], mask)
            ):
                raise ValueError(
                    f"Episodes for instruction {instruction!r} contain different text embeddings; "
                    f"first mismatch is {text_path}."
                )

            provenance = metadata.get("text_provenance")
            if provenance is None:
                legacy_episodes += 1
            else:
                if provenance.get("prompt_template") != DEFAULT_PROMPT:
                    raise ValueError(f"Prompt provenance mismatch in {metadata_path}.")
                if int(provenance.get("context_len", -1)) != current_context_len:
                    raise ValueError(f"Context-length provenance mismatch in {metadata_path}.")
                serialized_provenance = json.dumps(provenance, sort_keys=True)
                provenance_records.add(serialized_provenance)
                all_declared_provenance.add(serialized_provenance)

        entries_by_dataset[dataset_name] = dict(sorted(entries.items()))
        provenance_by_dataset[dataset_name] = {
            "episodes": len(metadata_paths),
            "legacy_episodes_without_provenance": legacy_episodes,
            "declared_provenance": [json.loads(item) for item in sorted(provenance_records)],
        }

    if len(all_declared_provenance) > 1:
        raise ValueError(
            "Selected LIBERO features declare multiple text encoders or tokenizers; "
            "build separate caches or re-extract with one text-conditioning protocol."
        )

    if verify_benchmark:
        benchmark_sets = _benchmark_instruction_sets(dataset_names)
        for dataset_name, entries in entries_by_dataset.items():
            extracted = set(entries)
            expected = benchmark_sets[dataset_name]
            if extracted != expected:
                raise ValueError(
                    f"Instruction mismatch for {dataset_name}: "
                    f"extra={sorted(extracted - expected)}, missing={sorted(expected - extracted)}."
                )

    payload = {
        "schema_version": TEXT_FEATURE_CACHE_SCHEMA_VERSION,
        "prompt_template": DEFAULT_PROMPT,
        "context_len": int(context_len),
        "text_dim": int(text_dim),
        "datasets": entries_by_dataset,
        "source_provenance": provenance_by_dataset,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    logger.info(
        "Saved %d LIBERO instruction embeddings to %s",
        sum(len(entries) for entries in entries_by_dataset.values()), output_path,
    )
    return payload


def main():
    args = parse_args()
    dataset_names = [name.strip() for name in args.dataset_names.split(",") if name.strip()]
    if not dataset_names:
        raise ValueError("--dataset_names must select at least one dataset.")
    output_path = args.output_path or args.features_root / TEXT_FEATURE_CACHE_FILENAME
    build_cache(
        args.features_root,
        dataset_names,
        output_path,
        verify_all_duplicates=args.verify_all_duplicates,
        verify_benchmark=args.verify_benchmark,
    )


if __name__ == "__main__":
    main()
