#!/usr/bin/env python
"""
run_libero_parallel_miniwam.py

Parallel execution manager for MiniWAM evaluation on LIBERO.
Dynamically schedules evaluation tasks across available GPUs.

Usage:
    python scripts/eval/run_libero_parallel_miniwam.py \
        --checkpoint outputs/libero_stage2/latest \
        --task_suite_names libero_spatial,libero_object \
        --gpu_ids 0,1,2 \
        --max_tasks_per_gpu 1 \
        --num_trials 50
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from gpu_environment import assign_egl_device, parse_gpu_capacity
from libero_plus_protocol import MAX_STEPS as COMMON_MAX_STEPS
from libero_eval_recovery import (
    atomic_write_json,
    atomic_write_text,
    summary_to_result,
    validate_complete_summary,
)

# Add LIBERO to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
libero_path = project_root / "LIBERO"
if libero_path.exists():
    sys.path.insert(0, str(libero_path))


def ensure_libero_config() -> Path:
    """Create a non-interactive LIBERO config so subprocesses never wait on input()."""
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", project_root / ".libero")).expanduser()
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return config_dir

    benchmark_root = project_root / "LIBERO" / "libero" / "libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"assets: {benchmark_root / 'assets'}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"benchmark_root: {benchmark_root}",
                f"datasets: {benchmark_root.parent / 'datasets'}",
                f"init_states: {benchmark_root / 'init_files'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_dir

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_libero_parallel_miniwam")


def required_protocol(suite_name: str, policy_seed: int) -> dict:
    """Protocol fields a reusable per-task summary must record (common LIBERO protocol)."""
    return {
        "protocol": "common_libero_v1",
        "max_steps": int(COMMON_MAX_STEPS[suite_name]),
        "env_seed": 0,
        "policy_seed": int(policy_seed),
        "num_steps_wait": 10,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel LIBERO Evaluation Manager for MiniWAM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="MiniWAM checkpoint folder (e.g. outputs/libero_stage2/latest)")
    parser.add_argument("--task_suite_names", type=str, default="libero_spatial",
                        help="Comma-separated task suite names (e.g., libero_10,libero_spatial,libero_object,libero_goal)")
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated GPU IDs to use (e.g., 0,1,2)")
    parser.add_argument("--max_tasks_per_gpu", type=str, default="1",
                        help="Maximum concurrent evaluation tasks per GPU: one number for all GPUs, "
                             "or one per entry of --gpu_ids (e.g. 18,6)")
    parser.add_argument("--oom_retries", type=int, default=2,
                        help="Retries per task after CUDA OOM; GPU concurrency is reduced each time")
    parser.add_argument("--num_trials", type=int, default=50,
                        help="Number of trials per task")
    parser.add_argument("--output_dir", type=str, default="./eval_results/libero",
                        help="Base output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for evaluation")
    parser.add_argument("--replan_steps", type=int, default=None,
                        help="Number of steps to execute before query")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Number of denoising steps for each replan query")
    parser.add_argument("--action_horizon", type=int, default=None,
                        help="Action prediction horizon")
    parser.add_argument("--dtype", type=str, default="bf16",
                        help="Inference precision (fp32, fp16, bf16)")
    parser.add_argument("--tokenizer_id", type=str, default="google/umt5-xxl",
                        help="HuggingFace tokenizer ID or local snapshot path")
    parser.add_argument(
        "--text_feature_cache", type=str, default=None,
        help="LIBERO instruction cache (default: libero_instruction_features.pt in the checkpoint folder)",
    )
    parser.add_argument(
        "--online_text_encoding", action="store_true",
        help="Force each worker to encode its instruction with UMT5-XXL.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.oom_retries < 0:
        raise ValueError("--oom_retries must be non-negative.")
    if args.num_trials < 1:
        raise ValueError("--num_trials must be at least 1.")
    if args.online_text_encoding and args.text_feature_cache:
        raise ValueError("--online_text_encoding cannot be combined with --text_feature_cache.")
    if args.text_feature_cache and not Path(args.text_feature_cache).expanduser().is_file():
        raise FileNotFoundError(f"LIBERO text feature cache does not exist: {args.text_feature_cache}")
    libero_config_dir = ensure_libero_config()
    logger.info("Using LIBERO_CONFIG_PATH=%s", libero_config_dir)

    # 1. Parse GPU IDs. Maps gpu_id -> how many tasks may run on it at once
    gpu_capacities = parse_gpu_capacity(
        args.gpu_ids, args.max_tasks_per_gpu, "--gpu_ids", "--max_tasks_per_gpu",
    )
    gpu_ids = list(gpu_capacities)
    logger.info("Using GPUs: %s | Max tasks per GPU: %s", gpu_ids, list(gpu_capacities.values()))

    # 2. Parse Task Suites
    suites = [s.strip() for s in args.task_suite_names.split(",") if s.strip()]
    if not suites:
        logger.error("No valid task suites provided.")
        sys.exit(1)
    unsupported = [suite for suite in suites if suite not in COMMON_MAX_STEPS]
    if unsupported:
        # Refuse up front, so a typo cannot leave a suite out of the report.
        raise ValueError(
            f"Unsupported LIBERO suite(s) {unsupported}; choose from {sorted(COMMON_MAX_STEPS)}."
        )

    # Import libero benchmark inside to get number of tasks
    # We do it dynamically to avoid importing it at module level if not installed
    try:
        from libero.libero import benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
    except ImportError:
        logger.error("Could not import libero. Please make sure the 'miniwam' environment is active.")
        sys.exit(1)

    # 3. Set up output and collect only tasks without a complete, compatible summary.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_queue = []
    recovered_results = []
    for suite_name in suites:
        if suite_name not in benchmark_dict:
            raise ValueError(f"LIBERO does not define the suite {suite_name!r}.")
        suite = benchmark_dict[suite_name]()
        n_tasks = int(suite.n_tasks)
        for task_id in range(n_tasks):
            task_output_dir = output_dir / suite_name / f"task_{task_id}"
            summary, reason = validate_complete_summary(
                task_output_dir / "summary.json",
                task_suite_name=suite_name,
                task_id=task_id,
                checkpoint=args.checkpoint,
                num_trials=args.num_trials,
                base_seed=args.seed,
                required_protocol=required_protocol(suite_name, args.seed),
            )
            if summary is not None:
                recovered_results.append(summary_to_result(summary))
                logger.info(
                    "[RECOVERED] Task %s:%d already has a valid %d-trial summary.",
                    suite_name, task_id, args.num_trials,
                )
            else:
                task_queue.append({"suite": suite_name, "task_id": task_id, "oom_attempt": 0})
                logger.info("[QUEUED] Task %s:%d | %s", suite_name, task_id, reason)

    total_tasks = len(task_queue) + len(recovered_results)
    if total_tasks == 0:
        raise ValueError("No valid LIBERO tasks were selected for evaluation.")
    logger.info(
        "Selected %d tasks across suites %s: %d complete, %d queued.",
        total_tasks, suites, len(recovered_results), len(task_queue),
    )

    # 5. Track GPU allocation
    # Maps gpu_id -> count of running tasks
    gpu_loads = {gpu_id: 0 for gpu_id in gpu_ids}

    # Active processes list: stores dicts of {"proc": subprocess.Popen, "gpu_id": int, "suite": str, "task_id": int, "output_path": Path}
    active_processes = []

    # Finished results summary list
    finished_results = list(recovered_results)

    start_time = time.time()

    # 6. Main execution loop
    while len(task_queue) > 0 or len(active_processes) > 0:
        # Check active processes for completion
        completed_procs = []
        for ap in active_processes:
            poll = ap["proc"].poll()
            if poll is not None:
                # Process completed!
                completed_procs.append(ap)
                gpu_loads[ap["gpu_id"]] -= 1
                ap["log_file"].close()

                summary, summary_error = validate_complete_summary(
                    ap["output_path"] / "summary.json",
                    task_suite_name=ap["suite"],
                    task_id=ap["task_id"],
                    checkpoint=args.checkpoint,
                    num_trials=args.num_trials,
                    base_seed=args.seed,
                    required_protocol=required_protocol(ap["suite"], args.seed),
                )
                if summary is not None:
                    result = summary_to_result(summary)
                    logger.info(
                        "[SUCCESS] Task %s:%d | %s | Success Rate: %.1f%% (%d/%d)",
                        ap["suite"], ap["task_id"], result["description"],
                        result["success_rate"], result["success_count"], args.num_trials,
                    )
                    finished_results.append(result)
                elif poll != 0:
                    log_text = ap["log_path"].read_text(errors="replace").lower()
                    is_oom = "out of memory" in log_text and (
                        "cuda" in log_text or "cublas" in log_text
                    )
                    is_resource_kill = poll in (-9, -15)
                    retryable = is_oom or is_resource_kill
                    if retryable and ap["oom_attempt"] < args.oom_retries:
                        gpu_capacities[ap["gpu_id"]] = max(
                            1, gpu_capacities[ap["gpu_id"]] - 1,
                        )
                        next_attempt = ap["oom_attempt"] + 1
                        task_queue.append({
                            "suite": ap["suite"],
                            "task_id": ap["task_id"],
                            "oom_attempt": next_attempt,
                        })
                        logger.warning(
                            "[RECOVERY RETRY] Task %s:%d rc=%d oom=%s attempt %d/%d; "
                            "GPU %d capacity is now %d.",
                            ap["suite"], ap["task_id"], poll, is_oom,
                            next_attempt, args.oom_retries,
                            ap["gpu_id"], gpu_capacities[ap["gpu_id"]],
                        )
                    else:
                        logger.error(
                            "[FAILED] Task %s:%d failed with return code %d (%s). Check %s.",
                            ap["suite"], ap["task_id"], poll, summary_error, ap["log_path"],
                        )
                        finished_results.append({
                            "suite": ap["suite"],
                            "task_id": ap["task_id"],
                            "description": f"task_{ap['task_id']}",
                            "success_rate": 0.0,
                            "success_count": 0,
                            "status": f"failed (rc={poll}, oom={is_oom}, summary={summary_error})"
                        })
                else:
                    logger.error(
                        "[FAILED] Task %s:%d exited successfully but %s.",
                        ap["suite"], ap["task_id"], summary_error,
                    )
                    finished_results.append({
                        "suite": ap["suite"],
                        "task_id": ap["task_id"],
                        "description": f"task_{ap['task_id']}",
                        "success_rate": 0.0,
                        "success_count": 0,
                        "status": f"failed ({summary_error})",
                    })

        # Remove completed from active processes list
        for cp in completed_procs:
            active_processes.remove(cp)

        # Launch new tasks if resources available
        while len(task_queue) > 0:
            # Find the least loaded GPU that has capacity
            available_gpu = None
            min_load = max(gpu_capacities.values()) + 1
            for gpu_id in gpu_ids:
                if gpu_loads[gpu_id] < gpu_capacities[gpu_id] and gpu_loads[gpu_id] < min_load:
                    min_load = gpu_loads[gpu_id]
                    available_gpu = gpu_id

            if available_gpu is None:
                # No GPU has capacity right now
                break

            # Launch next task
            task = task_queue.pop(0)
            suite_name = task["suite"]
            task_id = task["task_id"]
            oom_attempt = task["oom_attempt"]
            task_output_dir = output_dir / suite_name / f"task_{task_id}"
            task_output_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = task_output_dir / "evaluation.log"

            # Construct evaluation command
            cmd = [
                sys.executable,
                "scripts/eval/eval_libero_miniwam.py",
                "--checkpoint", args.checkpoint,
                "--task_suite_name", suite_name,
                "--task_id", str(task_id),
                "--num_trials", str(args.num_trials),
                "--gpu_id", str(available_gpu),
                "--output_dir", args.output_dir,
                "--seed", str(args.seed),
                "--dtype", args.dtype
            ]
            if args.replan_steps is not None:
                cmd.extend(["--replan_steps", str(args.replan_steps)])
            if args.num_inference_steps is not None:
                cmd.extend(["--num_inference_steps", str(args.num_inference_steps)])
            if args.action_horizon is not None:
                cmd.extend(["--action_horizon", str(args.action_horizon)])
            if args.tokenizer_id:
                cmd.extend(["--tokenizer_id", args.tokenizer_id])
            if args.text_feature_cache:
                cmd.extend(["--text_feature_cache", args.text_feature_cache])
            if args.online_text_encoding:
                cmd.append("--online_text_encoding")

            # Launch process and log stdout/stderr to log file
            log_file = open(log_file_path, "a")
            log_file.write(
                f"\n===== recovery attempt {oom_attempt} started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
            )
            log_file.flush()
            env = os.environ.copy()
            env["LIBERO_CONFIG_PATH"] = str(libero_config_dir)
            env["HF_HUB_OFFLINE"] = "1"
            assign_egl_device(env, available_gpu)
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )

            gpu_loads[available_gpu] += 1
            active_processes.append({
                "proc": proc,
                "gpu_id": available_gpu,
                "suite": suite_name,
                "task_id": task_id,
                "output_path": task_output_dir,
                "log_file": log_file,
                "log_path": log_file_path,
                "oom_attempt": oom_attempt,
            })

            logger.info("Launched task %s:%d on GPU %d (Log: %s)",
                        suite_name, task_id, available_gpu, log_file_path)

            # Stagger process spawns to prevent system RAM OOM spikes from simultaneous model loading
            time.sleep(15.0)

        # Sleep for a bit before checking again
        time.sleep(2.0)

    # 7. Print final results report
    total_time = time.time() - start_time
    logger.info("==================================================================")
    logger.info("Parallel Evaluation Complete! Total time: %.1f minutes", total_time / 60.0)
    logger.info("==================================================================")

    # Group results by suite
    finished_results.sort(key=lambda result: (result["suite"], result["task_id"]))
    results_by_suite = {}
    for res in finished_results:
        suite = res["suite"]
        if suite not in results_by_suite:
            results_by_suite[suite] = []
        results_by_suite[suite].append(res)

    # Calculate overall metrics
    all_rates = [r["success_rate"] for r in finished_results]
    overall_avg = sum(all_rates) / len(all_rates) if all_rates else 0.0
    total_tasks = len(finished_results)
    total_episodes = total_tasks * args.num_trials
    total_successes = sum([r.get("success_count", 0) for r in finished_results])
    total_failed_eps = total_episodes - total_successes
    failed_tasks = [r for r in finished_results if r["status"] != "success"]

    # Generate reports
    md_report = []
    md_report.append("# MiniWAM LIBERO Evaluation Report")
    md_report.append(f"- **Model Checkpoint**: `{args.checkpoint}`")
    md_report.append(f"- **Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_report.append(f"- **Total Duration**: {total_time / 60.0:.1f} minutes")
    md_report.append(f"- **Trials per Task**: {args.num_trials}")
    md_report.append(f"- **Total Tasks Evaluated**: {total_tasks} ({len(failed_tasks)} failed executions)")
    md_report.append(f"- **Total Episodes Evaluated**: {total_episodes} ({total_successes} successful, {total_failed_eps} failed)")
    md_report.append(f"- **Overall Average Success Rate**: {overall_avg:.1f}%")
    md_report.append("")

    for suite, suite_results in results_by_suite.items():
        suite_results = sorted(suite_results, key=lambda x: x["task_id"])
        rates = [r["success_rate"] for r in suite_results]
        avg_rate = sum(rates) / len(rates) if rates else 0.0

        md_report.append(f"## Suite: {suite} (Average Success: {avg_rate:.1f}%)")
        md_report.append("| Task ID | Task Description | Success Rate | Success Count | Status |")
        md_report.append("| --- | --- | --- | --- | --- |")
        for res in suite_results:
            md_report.append(f"| {res['task_id']} | {res['description']} | {res['success_rate']:.1f}% | {res['success_count']}/{args.num_trials} | {res['status']} |")
        md_report.append("")

        logger.info("Suite %s - Average Success Rate: %.1f%%", suite, avg_rate)
        for res in suite_results:
            logger.info("  Task %d: %s | Success Rate: %.1f%%", res["task_id"], res["description"], res["success_rate"])

    logger.info("=========================================================")
    logger.info("Overall Average Success Rate: %.1f%%", overall_avg)
    logger.info("Total Episodes: %d (%d succeeded, %d failed)", total_episodes, total_successes, total_failed_eps)
    logger.info("=========================================================")

    # Save markdown report
    report_md_path = output_dir / "report.md"
    atomic_write_text(report_md_path, "\n".join(md_report) + "\n")
    logger.info("Saved markdown report to %s", report_md_path)

    # Save json report
    report_json_path = output_dir / "report.json"
    atomic_write_json(report_json_path, {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "total_time_seconds": total_time,
        "timestamp": time.time(),
        "overall_metrics": {
            "average_success_rate": overall_avg,
            "total_tasks": total_tasks,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "total_failed_episodes": total_failed_eps,
            "execution_failures": len(failed_tasks)
        },
        "results": finished_results
    })
    logger.info("Saved JSON report to %s", report_json_path)
    if failed_tasks:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
