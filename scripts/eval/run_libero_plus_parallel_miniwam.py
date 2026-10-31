#!/usr/bin/env python3
"""Persistent multi-GPU MiniWAM evaluation on full or sampled LIBERO-Plus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from miniwam.libero_plus_text_cache import (  # noqa: E402
    LIBERO_PLUS_INSTRUCTION_CONTRACT,
    LiberoPlusTextFeatureCache,
    registry_manifest_id,
)
from miniwam.models.dinov3 import DINO_MINIWAM_FEATURE_CONTRACT  # noqa: E402
from scripts.eval.gpu_environment import (  # noqa: E402
    assign_egl_device,
    parse_gpu_capacity,
    worker_gpus,
)
from scripts.eval.libero_plus_protocol import (  # noqa: E402
    CLASSIFICATION_SHA256,
    LIBERO_PLUS_REV,
    MAX_STEPS,
    aggregate_results,
    load_registry,
    load_task_manifest,
    load_vanilla_instruction_registry,
    sha256_file,
    validate_manifest_against_registry,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="MiniWAM checkpoint folder")
    parser.add_argument("--libero_plus_root", type=Path, default=PROJECT_ROOT / "LIBERO-plus")
    parser.add_argument("--libero_root", type=Path, default=PROJECT_ROOT / "LIBERO")
    parser.add_argument("--text_feature_cache", type=Path, required=True)
    parser.add_argument("--task_manifest", type=Path, default=None, help="Omit for all 10,030 tasks.")
    parser.add_argument("--gpu_ids", default="0", help="Comma-separated GPU ids, e.g. 0,1.")
    parser.add_argument(
        "--max_tasks_per_gpu", default="1",
        help="Workers per GPU, each evaluating one task at a time: one value for all GPUs, "
             "or one per entry of --gpu_ids (e.g. 18,6).",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--launch_stagger_seconds", type=float, default=10.0)
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
    return parser.parse_args()


def _read_worker_results(worker_dir: Path) -> list[dict]:
    results = []
    for path in sorted(worker_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
    return results


def _next_attempt_index(worker_dir: Path) -> int:
    """Return an unused attempt index so resume never overwrites diagnostics."""
    indices = []
    for path in worker_dir.iterdir() if worker_dir.is_dir() else ():
        match = re.search(r"_attempt(\d+)(?:\.[^.]+)?$", path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def _protocol_id(protocol: dict) -> str:
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _visual_feature_contract(checkpoint: Path) -> str | None:
    import yaml

    config_path = checkpoint / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError(f"MiniWAM config must contain a mapping: {config_path}.")
    model_cfg = cfg.get("model", cfg)
    visual_cfg = model_cfg.get("visual_encoder", {}) or {}
    visual_type = str(visual_cfg.get("type", "wan_vae"))
    if visual_type == "dinov3":
        return DINO_MINIWAM_FEATURE_CONTRACT
    if visual_type != "wan_vae":
        raise ValueError(f"Unsupported visual encoder type {visual_type!r} in {config_path}.")
    return None


def _add_visual_feature_contract(protocol: dict, checkpoint: Path) -> None:
    contract = _visual_feature_contract(checkpoint)
    if contract is not None:
        protocol["visual_feature_contract"] = contract


def _latest_by_task(
    results: list[dict], *, checkpoint: Path, manifest_id: str, protocol_id: str
) -> dict[str, dict]:
    latest = {}
    for result in results:
        if result.get("benchmark_revision") != LIBERO_PLUS_REV:
            raise ValueError("Existing result uses a different LIBERO-Plus revision.")
        if result.get("classification_sha256") != CLASSIFICATION_SHA256:
            raise ValueError("Existing result uses a different classification registry.")
        if result.get("selection_manifest_id") != manifest_id:
            raise ValueError("Existing result uses a different selection manifest.")
        if result.get("protocol_id") != protocol_id:
            raise ValueError("Existing result uses a different evaluation protocol.")
        if Path(result.get("checkpoint", "")).resolve() != checkpoint.resolve():
            raise ValueError("Existing result belongs to another checkpoint.")
        key = f"{result['suite']}:{int(result['task_id'])}"
        latest[key] = result
    return latest


def _write_reports(output_dir: Path, results: list[dict], expected_tasks: int, protocol: dict):
    aggregate = aggregate_results(results)
    completed = aggregate.get("overall", {}).get("all", {"episodes": 0, "successes": 0, "success_rate": 0.0})
    failures = aggregate["execution_failures"]
    report = {
        "protocol": protocol,
        "expected_tasks": expected_tasks,
        "completed_tasks": completed["episodes"],
        "complete": completed["episodes"] == expected_tasks and not failures,
        "metrics": aggregate,
        "results": results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    lines = [
        "# MiniWAM LIBERO-Plus Evaluation",
        "",
        f"- Expected tasks: {expected_tasks}",
        f"- Completed tasks: {completed['episodes']}",
        f"- Execution failures: {len(failures)}",
        f"- Successes: {completed['successes']}",
        f"- Overall success rate: {100.0 * completed['success_rate']:.2f}%",
        f"- Standard protocol: {protocol['standard_protocol']}",
    ]
    for heading, key in (("Suite", "by_suite"), ("Category", "by_category"), ("Difficulty", "by_difficulty")):
        lines += [
            "",
            f"## By {heading.lower()}",
            "",
            f"| {heading} | Successes | Episodes | Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, counts in aggregate.get(key, {}).items():
            lines.append(
                f"| {name} | {counts['successes']} | {counts['episodes']} | "
                f"{100.0 * counts['success_rate']:.2f}% |"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    args = parse_args()
    if args.retries < 0 or args.launch_stagger_seconds < 0:
        raise ValueError("Retries and launch stagger must be non-negative.")
    # One GPU per worker; each worker loads the model once and runs its share of the tasks.
    worker_gpu_ids = worker_gpus(parse_gpu_capacity(
        args.gpu_ids, args.max_tasks_per_gpu, "--gpu_ids", "--max_tasks_per_gpu",
    ))
    registry = load_registry(args.libero_plus_root)
    # Validate the pinned vanilla checkout in the manager as well as each worker.
    # This fails before any output or GPU process is launched if Python would use
    # the wrong source for MiniWAM-compatible task language.
    load_vanilla_instruction_registry(args.libero_root)
    if args.task_manifest is None:
        records = registry
        manifest_id = registry_manifest_id(records)
        selection_name = "full_10030"
    else:
        records, metadata = load_task_manifest(args.task_manifest)
        validate_manifest_against_registry(records, registry)
        manifest_id = metadata["manifest_id"]
        selection_name = args.task_manifest.name

    cache = LiberoPlusTextFeatureCache(args.text_feature_cache)
    cache.assert_protocol(
        benchmark_revision=LIBERO_PLUS_REV,
        classification_sha256=CLASSIFICATION_SHA256,
        manifest_id=manifest_id,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = args.output_dir / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(worker_dir.glob("*.jsonl"))
    if existing_files and not args.resume:
        raise FileExistsError(
            f"Existing worker results found under {worker_dir}; pass --resume or use a new output directory."
        )

    protocol = {
        "benchmark_revision": LIBERO_PLUS_REV,
        "classification_sha256": CLASSIFICATION_SHA256,
        "selection": selection_name,
        "selection_manifest_id": manifest_id,
        "text_cache": str(args.text_feature_cache),
        "text_cache_sha256": sha256_file(args.text_feature_cache),
        "text_cache_manifest_id": cache.manifest_id,
        "vanilla_text_cache_sha256": cache.payload["text_provenance"][
            "vanilla_cache_sha256"
        ],
        "instruction_contract": LIBERO_PLUS_INSTRUCTION_CONTRACT,
        "checkpoint": str(args.checkpoint),
        "libero_plus_root": str(args.libero_plus_root),
        "libero_root": str(args.libero_root),
        "dtype": args.dtype,
        "vae_path": str(args.vae_path),
        "action_horizon_override": args.action_horizon,
        "replan_steps_override": args.replan_steps,
        "num_inference_steps_override": args.num_inference_steps,
        "policy_seed": args.policy_seed,
        "env_seed": args.env_seed,
        "num_steps_wait": args.num_steps_wait,
        "save_videos": args.save_videos,
        "video_sampling": (
            "sha256_task_key_mod10_v1" if args.save_videos == "sample10" else None
        ),
        "num_trials_per_task": 1,
        "initial_state_index": 0,
        "suite_max_policy_steps": MAX_STEPS,
        "standard_protocol": (
            args.policy_seed == 42 and args.env_seed == 0 and args.num_steps_wait == 10
        ),
    }
    _add_visual_feature_contract(protocol, args.checkpoint)
    protocol["protocol_id"] = _protocol_id(protocol)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    first_attempt = _next_attempt_index(worker_dir)
    for retry_index in range(args.retries + 1):
        attempt = first_attempt + retry_index
        previous = _read_worker_results(worker_dir)
        latest = (
            _latest_by_task(
                previous,
                checkpoint=args.checkpoint,
                manifest_id=manifest_id,
                protocol_id=protocol["protocol_id"],
            )
            if previous
            else {}
        )
        pending = [record for record in records if latest.get(record.key, {}).get("status") != "completed"]
        if not pending:
            break
        shards = [pending[index::len(worker_gpu_ids)] for index in range(len(worker_gpu_ids))]
        processes = []
        for worker, (gpu_id, shard) in enumerate(zip(worker_gpu_ids, shards)):
            if not shard:
                continue
            name = f"gpu{gpu_id}_w{worker:02d}"   # several workers may share a GPU
            task_path = worker_dir / f"tasks_{name}_attempt{attempt}.json"
            task_path.write_text(
                json.dumps([asdict(record) for record in shard], indent=2) + "\n", encoding="utf-8",
            )
            result_path = worker_dir / f"results_{name}_attempt{attempt}.jsonl"
            log_path = worker_dir / f"{name}_attempt{attempt}.log"
            cmd = [
                sys.executable,
                "scripts/eval/eval_libero_plus_miniwam.py",
                "--checkpoint", str(args.checkpoint),
                "--libero_plus_root", str(args.libero_plus_root),
                "--libero_root", str(args.libero_root),
                "--config_dir", str(args.output_dir / f"libero_config_{name}"),
                "--text_feature_cache", str(args.text_feature_cache),
                "--task_spec_json", str(task_path),
                "--manifest_id", manifest_id,
                "--protocol_id", protocol["protocol_id"],
                "--gpu_id", str(gpu_id),
                "--dtype", args.dtype,
                "--vae_path", str(args.vae_path),
                "--policy_seed", str(args.policy_seed),
                "--env_seed", str(args.env_seed),
                "--num_steps_wait", str(args.num_steps_wait),
                "--save_videos", args.save_videos,
                "--output_dir", str(args.output_dir),
                "--results_jsonl", str(result_path),
            ]
            optional = (
                ("--action_horizon", args.action_horizon),
                ("--replan_steps", args.replan_steps),
                ("--num_inference_steps", args.num_inference_steps),
            )
            for flag, value in optional:
                if value is not None:
                    cmd.extend((flag, str(value)))
            env = os.environ.copy()
            plus_root = str(args.libero_plus_root.expanduser().resolve())
            env["MINIWAM_LIBERO_ROOT"] = plus_root
            env["PYTHONPATH"] = plus_root + os.pathsep + env.get("PYTHONPATH", "")
            env["HF_HUB_OFFLINE"] = "1"
            assign_egl_device(env, gpu_id)
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
            )
            processes.append((process, log_handle, name, log_path))
            if args.launch_stagger_seconds:
                time.sleep(args.launch_stagger_seconds)
        for process, log_handle, name, log_path in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code:
                print(
                    f"Worker {name} attempt {attempt} exited {return_code}; see {log_path}.",
                    file=sys.stderr,
                )

    all_results = _read_worker_results(worker_dir)
    latest = _latest_by_task(
        all_results,
        checkpoint=args.checkpoint,
        manifest_id=manifest_id,
        protocol_id=protocol["protocol_id"],
    )
    ordered = [latest[record.key] for record in records if record.key in latest]
    report = _write_reports(args.output_dir, ordered, len(records), protocol)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
