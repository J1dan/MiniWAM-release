#!/usr/bin/env python3
"""Evaluate one or more LIBERO-Plus tasks in one persistent MiniWAM process."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from miniwam.libero_plus_text_cache import (  # noqa: E402
    LiberoPlusTextFeatureCache,
)
from scripts.eval.libero_plus_protocol import (  # noqa: E402
    CLASSIFICATION_SHA256,
    LIBERO_PLUS_REV,
    LiberoPlusTask,
    configure_libero_plus,
    instantiate_suite,
    load_registry,
    load_task_manifest,
    load_vanilla_instruction_registry,
    resolve_libero_plus_instructions,
    validate_manifest_against_registry,
)
from scripts.eval.gpu_environment import assign_egl_device  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="MiniWAM checkpoint folder")
    parser.add_argument("--libero_plus_root", type=Path, default=PROJECT_ROOT / "LIBERO-plus")
    parser.add_argument("--libero_root", type=Path, default=PROJECT_ROOT / "LIBERO")
    parser.add_argument("--config_dir", type=Path, default=PROJECT_ROOT / ".libero_plus")
    parser.add_argument("--text_feature_cache", type=Path, required=True)
    parser.add_argument("--task_manifest", type=Path, default=None)
    parser.add_argument("--task_spec_json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--manifest_id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--protocol_id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task_suite_name", default=None)
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--vae_path", type=Path, default=PROJECT_ROOT / "pretrained/vision/Wan2.1_VAE.pth")
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--replan_steps", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--policy_seed", type=int, default=42)
    parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument(
        "--save_videos", choices=("none", "failures", "sample10", "all"), default="none",
    )
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "eval_results/libero_plus")
    parser.add_argument("--results_jsonl", type=Path, default=None)
    return parser.parse_args()


def _record_from_dict(value: dict) -> LiberoPlusTask:
    return LiberoPlusTask(
        suite=value["suite"],
        task_id=int(value["task_id"]),
        official_id=int(value["official_id"]),
        task_name=value["task_name"],
        category=value["category"],
        difficulty=value.get("difficulty"),
    )


def _selected_records(args, registry):
    if args.task_spec_json is not None:
        values = json.loads(args.task_spec_json.read_text(encoding="utf-8"))
        records = [_record_from_dict(value) for value in values]
        manifest_id = args.manifest_id
    elif args.task_manifest is not None:
        records, metadata = load_task_manifest(args.task_manifest)
        manifest_id = metadata["manifest_id"]
    elif args.task_suite_name is not None and args.task_id is not None:
        by_key = {record.key: record for record in registry}
        key = f"{args.task_suite_name}:{args.task_id}"
        if key not in by_key:
            raise KeyError(f"Unknown LIBERO-Plus task {key}.")
        records = [by_key[key]]
        manifest_id = None
    else:
        raise ValueError("Select --task_manifest, --task_spec_json, or one suite/task ID.")
    validate_manifest_against_registry(records, registry)
    return records, manifest_id


def _append_result(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    args = parse_args()
    assign_egl_device(os.environ, args.gpu_id)
    os.environ["MINIWAM_LIBERO_ROOT"] = str(args.libero_plus_root.expanduser().resolve())
    registry = load_registry(args.libero_plus_root)
    vanilla_instruction_registry = load_vanilla_instruction_registry(args.libero_root)
    records, selected_manifest_id = _selected_records(args, registry)
    benchmark = configure_libero_plus(
        args.libero_plus_root, config_dir=args.config_dir, require_assets=True,
    )
    cache = LiberoPlusTextFeatureCache(args.text_feature_cache)
    cache.assert_protocol(
        benchmark_revision=LIBERO_PLUS_REV,
        classification_sha256=CLASSIFICATION_SHA256,
        manifest_id=selected_manifest_id,
    )

    from scripts.eval.miniwam_libero_plus_runtime import MiniWAMLiberoPlusRuntime

    runtime = MiniWAMLiberoPlusRuntime(
        checkpoint=args.checkpoint,
        gpu_id=args.gpu_id,
        dtype=args.dtype,
        vae_path=args.vae_path,
        action_horizon=args.action_horizon,
        replan_steps=args.replan_steps,
        num_inference_steps=args.num_inference_steps,
        policy_seed=args.policy_seed,
        env_seed=args.env_seed,
        num_steps_wait=args.num_steps_wait,
        save_videos=args.save_videos,
        output_dir=args.output_dir,
        vanilla_instruction_registry=vanilla_instruction_registry,
    )
    suites = {
        suite: instantiate_suite(benchmark, suite)
        for suite in {record.suite for record in records}
    }
    results_path = args.results_jsonl or args.output_dir / "results.jsonl"
    had_failure = False
    for record in records:
        task = suites[record.suite].get_task(record.task_id)
        instructions = resolve_libero_plus_instructions(
            record, task, vanilla_instruction_registry,
        )
        try:
            result = runtime.evaluate(record, suites[record.suite], cache)
        except Exception as exc:
            had_failure = True
            result = {
                "suite": record.suite,
                "task_id": record.task_id,
                "official_id": record.official_id,
                "task_name": record.task_name,
                "category": record.category,
                "difficulty": record.difficulty,
                "status": "execution_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        result.update(
            {
                # Keep both the fork's exact raw value and the instruction that
                # was actually encoded, including on execution failures.
                "task_language_raw": instructions.raw,
                "task_language_clean": instructions.clean,
                "task_language_base": instructions.base,
                "instruction_source": instructions.source,
                "checkpoint": str(args.checkpoint),
                "benchmark_revision": LIBERO_PLUS_REV,
                "classification_sha256": CLASSIFICATION_SHA256,
                "selection_manifest_id": selected_manifest_id,
                "text_cache_manifest_id": cache.manifest_id,
                "protocol_id": args.protocol_id,
                "policy_seed": args.policy_seed,
                "env_seed": args.env_seed,
                "num_steps_wait": args.num_steps_wait,
            }
        )
        _append_result(results_path, result)
        if result["status"] == "execution_failed" and "out of memory" in result["error"].lower():
            raise SystemExit(42)

    if len(records) == 1:
        summary_path = args.output_dir / record.suite / f"task_{record.task_id}" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
