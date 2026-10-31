#!/usr/bin/env python3
"""Validate official RoboTwin multitask outputs and compute leaderboard means."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

MINIWAM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MINIWAM_ROOT))

from miniwam.robotwin import ROBOTWIN_TASKS
from miniwam.robotwin_manifest import load_manifest, sha256, case_key


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FINAL_RATE = re.compile(r"Final(?: batch)? success rate:\s*(\d+)\s*/\s*(\d+)", re.I)
CHECKPOINT_SHA256 = re.compile(r"\[MiniWAM\] checkpoint_sha256=([0-9a-f]{64})")
STATISTICS_SHA256 = re.compile(r"\[MiniWAM\] dataset_stats_sha256=([0-9a-f]{64})")
EVALUATION_PROTOCOL = re.compile(r"\[MiniWAM\] evaluation_protocol=(frozen|official)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-summary", type=Path, required=True)
    parser.add_argument("--randomized-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/robotwin/metrics"))
    parser.add_argument("--expected-trials", type=int, default=100)
    parser.add_argument("--evaluation-protocol", choices=("frozen", "official"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--text-index", type=Path)
    return parser.parse_args()


def _job_log(summary_path: Path, job: dict[str, Any]) -> Path:
    configured = Path(str(job.get("log", "")))
    candidates = (
        configured,
        summary_path.parent / "logs" / f"{job['job_id']}.log",
        summary_path.parent / configured,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No official evaluator log found for {job.get('job_id')}: {candidates}")


def _checkpoint_label(job: dict[str, Any]) -> str:
    command = job.get("command")
    # Local multitask command: bash <policy/eval.sh> bench task checkpoint ...
    if isinstance(command, list) and len(command) >= 5:
        return str(command[4])
    raise ValueError(f"Job {job.get('job_id')} does not record its evaluator command/checkpoint.")


def frozen_provenance(manifest_path=None, index_path=None) -> dict[str, str]:
    """Revalidate and expose the sealed language/case provenance in aggregates."""
    manifest_value = manifest_path or os.environ.get("MINIWAM_EVAL_MANIFEST")
    index_value = index_path or os.environ.get("MINIWAM_FROZEN_TEXT_INDEX")
    if not manifest_value or not index_value:
        raise ValueError("Frozen aggregation requires --manifest and --text-index.")
    manifest = Path(manifest_value).expanduser().resolve()
    index_path = Path(index_value).expanduser().resolve()
    cases, provenance = load_manifest(manifest, MINIWAM_ROOT / "RoboTwin")
    from scripts.data.package_robotwin_eval_cache import validate_v2_certification
    from scripts.eval.robotwin_split_contract import BENCHMARK_PROTOCOL, require_marked_contract
    if provenance.get('benchmark_protocol') != BENCHMARK_PROTOCOL:
        raise ValueError('Frozen results must come from the certified official-split release')
    validate_v2_certification(provenance)
    require_marked_contract(cases, provenance, MINIWAM_ROOT/'RoboTwin')
    index = json.loads(index_path.read_text(encoding="utf-8"))
    seal = json.loads(index_path.with_suffix(".sha256.json").read_text(encoding="utf-8"))
    index_sha256 = sha256(index_path)
    if seal.get("text_cache_sha256") != index_sha256:
        raise ValueError("Frozen text-cache index SHA256 mismatch during aggregation.")
    if index.get("eval_manifest_sha256") != provenance.get("eval_manifest_sha256"):
        raise ValueError("Frozen text cache belongs to a different evaluation manifest.")
    return {
        "eval_manifest_sha256": provenance["eval_manifest_sha256"],
        "text_cache_sha256": index_sha256,
        "prism_commit": provenance["prism_commit"],
        "robotwin_commit": provenance["sources"]["robotwin_commit"],
        "xpolicylab_commit": provenance["sources"]["xpolicylab_commit"],
    }


def schedule_provenance(summary_path: Path) -> dict[str, str]:
    plan_path = Path(summary_path).expanduser().resolve().parent/'schedule_plan.json'
    plan = json.loads(plan_path.read_text())
    result = {key: plan.get(key) for key in ('robotwin_commit', 'xpolicylab_commit')}
    if any(not isinstance(value, str) or not value for value in result.values()):
        raise ValueError(f'Schedule plan has incomplete upstream provenance: {plan_path}')
    return result


def load_setting(
    summary_path: Path,
    *,
    expected_setting: str,
    expected_trials: int,
    expected_protocol: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    summary_path = summary_path.expanduser().resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"{summary_path} has no jobs list.")
    if payload.get("jobs_total") != 50 or payload.get("jobs_succeeded") != 50:
        raise ValueError(
            f"{summary_path} is not a complete run of the 50 benchmark tasks "
            f"(jobs total={payload.get('jobs_total')}, succeeded={payload.get('jobs_succeeded')}); "
            "the summary needs every task."
        )
    if expected_protocol is not None:
        plan = json.loads((summary_path.parent/'schedule_plan.json').read_text())
        contract = (plan.get('miniwam') or {}).get('benchmark_validation') or {}
        if (contract.get('benchmark_protocol') != 'official_split_v1' or
                contract.get('split_configs', {}).get(expected_setting, {}).get('instruction_type') !=
                {'demo_clean': 'seen', 'demo_randomized': 'unseen'}[expected_setting]):
            raise ValueError('Schedule lacks official instruction-split validation')
    rows = []
    checkpoints = set()
    checkpoint_hashes = set()
    statistics_hashes = set()
    seen_tasks = set()
    for job in jobs:
        if job.get("status") != "success":
            raise ValueError(f"Official evaluator job failed: {job}")
        task = str(job.get("task"))
        if task in seen_tasks:
            raise ValueError(f"Task {task!r} occurs more than once in {summary_path}.")
        seen_tasks.add(task)
        setting = str(job.get("overrides", {}).get("task_config"))
        if setting != expected_setting:
            raise ValueError(f"Task {task} used task_config={setting!r}, expected {expected_setting!r}.")
        instruction = job.get('overrides', {}).get('instruction_type')
        native_instruction = {'demo_clean': 'seen', 'demo_randomized': 'unseen'}[setting]
        if instruction is not None and instruction != native_instruction:
            raise ValueError(f'Task {task} used the wrong instruction split: {instruction}')
        text = ANSI_ESCAPE.sub("", _job_log(summary_path, job).read_text(encoding="utf-8"))
        if expected_protocol is not None:
            protocols = set(EVALUATION_PROTOCOL.findall(text))
            if protocols != {expected_protocol}:
                raise ValueError(
                    f"Task {task} did not prove evaluation_protocol={expected_protocol}: "
                    f"{sorted(protocols)}"
                )
            if expected_protocol == "official":
                if job.get("overrides", {}).get("expert_check") is not True:
                    raise ValueError(f"Official task {task} did not enable expert filtering.")
                if "MINIWAM_CASE_MODE=replay" in text:
                    raise ValueError(f"Official task {task} contains frozen replay evidence.")
        matches = FINAL_RATE.findall(text)
        if not matches:
            raise ValueError(f"Task {task} log has no final official success-rate line.")
        successes, trials = (int(value) for value in matches[-1])
        if trials != expected_trials or not 0 <= successes <= trials:
            raise ValueError(
                f"Task {task} counted {successes}/{trials}; expected 0..{expected_trials}/{expected_trials}."
            )
        checkpoint = _checkpoint_label(job)
        checkpoints.add(checkpoint)
        hash_matches = CHECKPOINT_SHA256.findall(text)
        if len(set(hash_matches)) != 1:
            raise ValueError(
                f"Task {task} log must record exactly one resolved MiniWAM checkpoint SHA256."
            )
        checkpoint_hash = hash_matches[-1]
        checkpoint_hashes.add(checkpoint_hash)
        statistics_matches = STATISTICS_SHA256.findall(text)
        if len(set(statistics_matches)) != 1:
            raise ValueError(
                f"Task {task} log must record exactly one resolved MiniWAM statistics SHA256."
            )
        statistics_hash = statistics_matches[-1]
        statistics_hashes.add(statistics_hash)
        rows.append({
            "task": task,
            "setting": setting,
            "successes": successes,
            "trials": trials,
            "success_rate": successes / trials,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_hash,
            "statistics_sha256": statistics_hash,
            "log": str(_job_log(summary_path, job)),
        })
    if seen_tasks != set(ROBOTWIN_TASKS):
        raise ValueError(
            f"Task set mismatch: missing={sorted(set(ROBOTWIN_TASKS) - seen_tasks)}, "
            f"extra={sorted(seen_tasks - set(ROBOTWIN_TASKS))}."
        )
    if len(checkpoints) != 1:
        raise ValueError(f"A setting used multiple checkpoints: {sorted(checkpoints)}")
    if len(checkpoint_hashes) != 1:
        raise ValueError(f"A setting resolved multiple checkpoint files: {sorted(checkpoint_hashes)}")
    if len(statistics_hashes) != 1:
        raise ValueError(f"A setting resolved multiple statistics files: {sorted(statistics_hashes)}")
    checkpoint_identity = f"{checkpoints.pop()}@sha256:{checkpoint_hashes.pop()}"
    return sorted(rows, key=lambda row: ROBOTWIN_TASKS.index(row["task"])), checkpoint_identity


def validate_setting_evidence(summary_path, rows, cases, provenance, expected_trials):
    """Bind reported scores to the actual completed frozen cases, not current env vars."""
    canonical = {case_key(case): case for case in cases}
    for row in rows:
        log_path = Path(row['log'])
        pointer = summary_path.parent/'jobs'/f'{log_path.stem}.artifacts.json'
        if pointer.is_file():
            directory = Path(json.loads(pointer.read_text())['path'])
        else:
            text = ANSI_ESCAPE.sub('', log_path.read_text())
            paths = re.findall(r'^Data has been saved to (.+)$', text, re.M)
            if not paths:
                raise ValueError(f'No frozen result evidence for {row["task"]}: {log_path}')
            result_path = Path(paths[-1].strip())
            if not result_path.is_absolute():
                result_path = MINIWAM_ROOT/'RoboTwin'/result_path
            directory = result_path.parent
        report = json.loads((directory/'summary.json').read_text())
        expected_provenance = {key: provenance[key] for key in (
            'eval_manifest_sha256', 'text_cache_sha256', 'robotwin_commit', 'xpolicylab_commit')}
        expected_provenance.update(checkpoint_sha256=row['checkpoint_sha256'],
                                   dataset_stats_sha256=row['statistics_sha256'])
        if any(report.get(key) != value for key, value in expected_provenance.items()):
            raise ValueError(f'Frozen result provenance mismatch: {directory}')
        results = report['results']
        expected_keys = {key for key, case in canonical.items() if key[:2] == (row['task'], row['setting'])
                         and case['episode_index'] < expected_trials}
        if (len(results) != expected_trials or {case_key(case) for case in results} != expected_keys):
            raise ValueError(f'Frozen result case membership/count mismatch: {directory}')
        for case in results:
            if (type(case.get('success')) is not bool or
                    any(case.get(key) != value for key, value in canonical[case_key(case)].items()) or
                    case.get('provenance') != expected_provenance):
                raise ValueError(f'Frozen result case/provenance mismatch: {directory}')
        successes = sum(case['success'] for case in results)
        if (successes != row['successes'] or report.get('total_successes') != successes or
                report.get('total_episodes') != expected_trials):
            raise ValueError(f'Official log and frozen result counts differ: {directory}')


def main() -> None:
    args = parse_args()
    if args.expected_trials <= 0:
        raise ValueError("--expected-trials must be positive.")
    clean, clean_checkpoint = load_setting(
        args.clean_summary,
        expected_setting="demo_clean",
        expected_trials=args.expected_trials,
        expected_protocol=args.evaluation_protocol,
    )
    randomized, randomized_checkpoint = load_setting(
        args.randomized_summary,
        expected_setting="demo_randomized",
        expected_trials=args.expected_trials,
        expected_protocol=args.evaluation_protocol,
    )
    if clean_checkpoint != randomized_checkpoint:
        raise ValueError(
            "Clean and randomized evaluations used different checkpoint labels: "
            f"{clean_checkpoint!r} vs {randomized_checkpoint!r}."
        )
    clean_statistics = {row["statistics_sha256"] for row in clean}
    randomized_statistics = {row["statistics_sha256"] for row in randomized}
    if clean_statistics != randomized_statistics:
        raise ValueError(
            "Clean and randomized evaluations used different normalization statistics: "
            f"{clean_statistics} vs {randomized_statistics}."
        )
    by_randomized = {row["task"]: row for row in randomized}
    task_rows = []
    for clean_row in clean:
        random_row = by_randomized[clean_row["task"]]
        task_rows.append({
            "task": clean_row["task"],
            "sr_clean": clean_row["success_rate"],
            "sr_randomized": random_row["success_rate"],
            "sr_avg": (clean_row["success_rate"] + random_row["success_rate"]) / 2,
            "clean_successes": clean_row["successes"],
            "randomized_successes": random_row["successes"],
            "trials_per_setting": args.expected_trials,
        })
    sr_clean = sum(row["sr_clean"] for row in task_rows) / 50
    sr_randomized = sum(row["sr_randomized"] for row in task_rows) / 50
    if args.evaluation_protocol == 'frozen':
        if args.manifest is None or args.text_index is None:
            raise ValueError('Frozen aggregation requires --manifest and --text-index.')
        provenance = frozen_provenance(args.manifest, args.text_index)
        cases, _ = load_manifest(args.manifest, MINIWAM_ROOT/'RoboTwin')
        validate_setting_evidence(args.clean_summary.resolve(), clean, cases, provenance, args.expected_trials)
        validate_setting_evidence(args.randomized_summary.resolve(), randomized, cases, provenance, args.expected_trials)
    else:
        if args.manifest is not None or args.text_index is not None:
            raise ValueError('Official aggregation must not accept frozen manifest/cache assets.')
        clean_provenance = schedule_provenance(args.clean_summary)
        randomized_provenance = schedule_provenance(args.randomized_summary)
        if clean_provenance != randomized_provenance:
            raise ValueError('Official clean/randomized runs use different upstream revisions.')
        provenance = clean_provenance
    statistics_sha256 = clean_statistics.pop()
    result = {
        "schema_version": 1,
        "evaluation_protocol": args.evaluation_protocol,
        "checkpoint": clean_checkpoint,
        "checkpoint_sha256": clean[0]["checkpoint_sha256"],
        "dataset_stats_sha256": statistics_sha256,
        "statistics_sha256": statistics_sha256,
        **provenance,
        "tasks": 50,
        "trials_per_task_per_setting": args.expected_trials,
        "full_benchmark_trial_count": args.expected_trials == 100,
        "counted_episodes": 100 * args.expected_trials,
        "SR_clean": sr_clean,
        "SR_randomized": sr_randomized,
        "SR_avg": (sr_clean + sr_randomized) / 2,
        "per_task": task_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8",
    )
    with (args.output_dir / "per_task.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    print(json.dumps({key: result[key] for key in ("SR_clean", "SR_randomized", "SR_avg")}, indent=2))


if __name__ == "__main__":
    main()
