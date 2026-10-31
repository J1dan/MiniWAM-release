#!/usr/bin/env python3
"""Audit every frozen scene, one isolated official task/condition block at a time.

The parent process schedules block workers in the RoboTwin environment.  A worker
does setup only (no expert and no policy rollout), applies the release-v1 asset
replay shim used by evaluation, and requires the reconstructed scene SHA256 to
equal the manifest SHA256 exactly. V2 cases may explicitly canonicalize the
documented nonphysical ``models.last_actor`` planner alias.
"""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from scripts.eval.gpu_environment import parse_gpu_capacity, worker_gpus
from scripts.eval.robotwin_replay_state import normalize_snapshot


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def difference_paths(expected, actual, prefix='', limit=200):
    """Return bounded structural/value difference paths without huge payloads."""
    result = []

    def visit(left, right, path):
        if len(result) >= limit:
            return
        if type(left) is not type(right):
            result.append(path or '$')
        elif isinstance(left, dict):
            for key in sorted(set(left) | set(right)):
                child = f'{path}.{key}' if path else key
                if key not in left or key not in right:
                    result.append(child)
                else:
                    visit(left[key], right[key], child)
                if len(result) >= limit:
                    return
        elif isinstance(left, list):
            if len(left) != len(right):
                result.append(f'{path}.length')
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f'{path}[{index}]')
                if len(result) >= limit:
                    return
        elif left != right:
            result.append(path or '$')

    visit(expected, actual, prefix)
    return result


def normalized_reconstructed_snapshot(case, snapshot):
    normalization = case.get('scene_normalization')
    if normalization is None:
        return snapshot, None
    return normalize_snapshot(case, snapshot), normalization


def audit_block(args):
    robotwin_root = args.robotwin_root.resolve()
    os.chdir(robotwin_root)
    sys.path.insert(0, str(robotwin_root))
    official = load_module('official_eval', robotwin_root/'scripts/eval_policy_xpolicylab.py')
    protocol = load_module('manifest_protocol', ROOT/'miniwam/robotwin_manifest.py')
    hooks = load_module(
        'frozen_hooks',
        ROOT/'integrations/xpolicylab/policy/MiniWAM/compat/frozen_cases.py',
    )
    cases, provenance = protocol.load_manifest(args.manifest.resolve(), robotwin_root)
    selected = sorted(
        (case for case in cases if case['task_name'] == args.worker_task and
         case['task_config'] == args.worker_setting),
        key=lambda case: case['episode_index'],
    )
    if len(selected) != 100 or [case['episode_index'] for case in selected] != list(range(100)):
        raise ValueError('Worker requires one complete 100-case task/condition block')
    cfg, _ = official.load_task_args(dict(
        task_name=args.worker_task,
        task_config=args.worker_setting,
        policy_name='SCENE_AUDIT',
    ))
    cfg['eval_mode'] = True
    env = official.class_decorator(args.worker_task)
    records = []
    report_path = args.worker_report.resolve()
    audit_tool_sha256 = protocol.sha256(__file__)
    scene_protocol_sha256 = protocol.sha256(ROOT/'miniwam/robotwin_manifest.py')
    replay_hook_sha256 = protocol.sha256(
        ROOT/'integrations/xpolicylab/policy/MiniWAM/compat/frozen_cases.py'
    )

    def save(complete=False):
        exact = sum(record['status'] == 'exact' for record in records)
        mismatches = sum(record['status'] == 'scene_mismatch' for record in records)
        errors = sum(record['status'] == 'setup_error' for record in records)
        protocol.atomic_json(report_path, dict(
            schema_version=1,
            manifest_sha256=provenance['eval_manifest_sha256'],
            audit_tool_sha256=audit_tool_sha256,
            scene_protocol_sha256=scene_protocol_sha256,
            replay_hook_sha256=replay_hook_sha256,
            replay_state_sha256=protocol.sha256(ROOT/'scripts/eval/robotwin_replay_state.py'),
            task_name=args.worker_task,
            task_config=args.worker_setting,
            expected_cases=100,
            checked_cases=len(records),
            exact_cases=exact,
            scene_mismatch_count=mismatches,
            setup_error_count=errors,
            complete=bool(complete and len(records) == 100 and errors == 0),
            cases=records,
        ))

    for case in selected:
        snapshot = None
        replay = None
        try:
            replay = hooks.ReleaseModelChoiceReplay(case, robotwin_root)
            replay.install()
            setup_complete = False
            try:
                env.setup_demo(
                    now_ep_num=case['episode_index'],
                    seed=case['accepted_seed'],
                    is_test=True,
                    **cfg,
                )
                setup_complete = True
            finally:
                replay.restore(require_complete=setup_complete)
            snapshot = protocol.scene_snapshot(
                env, case['scene_snapshot']['episode_info']
            )
            compared_snapshot, normalization = normalized_reconstructed_snapshot(case, snapshot)
            reconstructed = protocol.content_hash(compared_snapshot)
            if reconstructed == case['scene_sha256']:
                record = dict(
                    episode_index=case['episode_index'],
                    accepted_seed=case['accepted_seed'],
                    status='exact',
                    scene_sha256=reconstructed,
                )
                if normalization:
                    record['scene_normalization'] = normalization
            else:
                record = dict(
                    episode_index=case['episode_index'],
                    accepted_seed=case['accepted_seed'],
                    status='scene_mismatch',
                    expected_scene_sha256=case['scene_sha256'],
                    reconstructed_scene_sha256=reconstructed,
                    difference_paths=difference_paths(
                        case['scene_snapshot'], compared_snapshot
                    ),
                )
        except Exception as exc:
            if replay is not None:
                try:
                    replay.restore(require_complete=False)
                except Exception:
                    pass
            record = dict(
                episode_index=case['episode_index'],
                accepted_seed=case['accepted_seed'],
                status='setup_error',
                error_type=type(exc).__name__,
                error=str(exc),
            )
        finally:
            official.safe_close_env(env, clear_cache=True)
        records.append(record)
        save()
        print(
            f"[{len(records):03d}/100] {args.worker_task}/{args.worker_setting} "
            f"seed={case['accepted_seed']} {record['status']}",
            flush=True,
        )
    save(complete=True)
    if any(record['status'] == 'setup_error' for record in records):
        raise RuntimeError('Block audit contains setup errors; refusing a complete audit')


def report_is_complete(path, manifest_sha256, task, setting):
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    audit_hash, scene_hash, hook_hash = protocol_hashes()
    return (
        report.get('complete') is True and
        report.get('manifest_sha256') == manifest_sha256 and
        report.get('audit_tool_sha256') == audit_hash and
        report.get('scene_protocol_sha256') == scene_hash and
        report.get('replay_hook_sha256') == hook_hash and
        report.get('replay_state_sha256') == replay_state_hash() and
        report.get('task_name') == task and
        report.get('task_config') == setting and
        report.get('checked_cases') == 100 and
        report.get('setup_error_count') == 0 and
        report.get('exact_cases', 0) + report.get('scene_mismatch_count', 0) == 100 and
        len(report.get('cases', [])) == 100
    )


def protocol_hashes():
    import hashlib

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return (
        digest(__file__),
        digest(ROOT/'miniwam/robotwin_manifest.py'),
        digest(ROOT/'integrations/xpolicylab/policy/MiniWAM/compat/frozen_cases.py'),
    )


def replay_state_hash():
    import hashlib
    return hashlib.sha256((ROOT/'scripts/eval/robotwin_replay_state.py').read_bytes()).hexdigest()


def schedule(args):
    protocol = load_module('manifest_protocol_parent', ROOT/'miniwam/robotwin_manifest.py')
    _, provenance = protocol.load_manifest(args.manifest.resolve())
    tasks = yaml.safe_load(args.tasks_config.read_text())['tasks']
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError('Expected exactly 50 unique tasks')
    settings = (
        ('demo_clean', 'demo_randomized')
        if args.task_config == 'both' else (args.task_config,)
    )
    jobs = [(task, setting) for task in tasks for setting in settings]
    output = args.output_dir.resolve()
    blocks_dir = output/'blocks'
    logs_dir = output/'logs'
    blocks_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    capacity = parse_gpu_capacity(args.gpu_ids, args.jobs_per_gpu, '--gpu-ids', '--jobs-per-gpu')
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)

    def run_slot(gpu):
        # Each slot takes the next block when it is free, so slow blocks do not hold up others.
        while True:
            try:
                task, setting = pending.get_nowait()
            except queue.Empty:
                return
            report = blocks_dir/f'{task}__{setting}.json'
            if report_is_complete(report, provenance['eval_manifest_sha256'], task, setting):
                print(f'[SKIP] {task}/{setting}', flush=True)
                continue
            command = [
                'conda', 'run', '--no-capture-output', '-p', str(args.env_prefix.resolve()),
                'python', str(Path(__file__).resolve()),
                '--manifest', str(args.manifest.resolve()),
                '--robotwin-root', str(args.robotwin_root.resolve()),
                '--worker-task', task,
                '--worker-setting', setting,
                '--worker-report', str(report),
            ]
            if args.dry_run:
                print(json.dumps(dict(gpu=gpu, command=command)), flush=True)
                continue
            print(f'[AUDIT] {task}/{setting} gpu={gpu}', flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       PYTHONPATH=f'{ROOT}:{os.environ.get("PYTHONPATH", "")}')
            with (logs_dir/f'{task}__{setting}.log').open('w') as stream:
                subprocess.run(command, cwd=args.robotwin_root, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)

    slots = worker_gpus(capacity)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        list(pool.map(run_slot, slots))
    if args.dry_run:
        return

    reports = []
    for task, setting in jobs:
        path = blocks_dir/f'{task}__{setting}.json'
        if not report_is_complete(path, provenance['eval_manifest_sha256'], task, setting):
            raise ValueError(f'Incomplete block audit: {task}/{setting}')
        reports.append(json.loads(path.read_text()))
    checked = sum(report['checked_cases'] for report in reports)
    expected = 100 * len(jobs)
    mismatch_blocks = [
        dict(task_name=report['task_name'], task_config=report['task_config'],
             scene_mismatch_count=report['scene_mismatch_count'])
        for report in reports if report['scene_mismatch_count']
    ]
    audit = dict(
        schema_version=1,
        complete=checked == expected,
        manifest_sha256=provenance['eval_manifest_sha256'],
        audit_tool_sha256=protocol_hashes()[0],
        replay_state_sha256=replay_state_hash(),
        scene_protocol_sha256=protocol_hashes()[1],
        replay_hook_sha256=protocol_hashes()[2],
        settings=list(settings),
        expected_blocks=len(jobs),
        audited_blocks=len(reports),
        expected_cases=expected,
        checked_cases=checked,
        exact_cases=sum(report['exact_cases'] for report in reports),
        scene_mismatch_cases=sum(report['scene_mismatch_count'] for report in reports),
        setup_error_cases=sum(report['setup_error_count'] for report in reports),
        mismatch_blocks=mismatch_blocks,
        block_reports={
            f"{report['task_name']}/{report['task_config']}":
                str((blocks_dir/f"{report['task_name']}__{report['task_config']}.json").relative_to(output))
            for report in reports
        },
    )
    protocol.atomic_json(output/'audit.json', audit)
    protocol.atomic_json(output/'recapture_plan.json', dict(
        schema_version=1,
        complete=audit['complete'] and audit['setup_error_cases'] == 0,
        source_manifest_sha256=provenance['eval_manifest_sha256'],
        audited_settings=list(settings),
        audited_cases=checked,
        blocks=mismatch_blocks,
    ))
    if audit['exact_cases'] != expected:
        # A machine that does not rebuild every recorded scene cannot replay the frozen cases.
        print(
            f"FAIL: {audit['exact_cases']}/{expected} cases rebuilt exactly; "
            f"{len(mismatch_blocks)} blocks require recapture (see {output/'audit.json'})",
            file=sys.stderr, flush=True,
        )
        raise SystemExit(1)
    print(f"PASS: all {expected} cases rebuilt exactly", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT/'datasets/robotwin_eval_bundle_v3_unseen/robotwin_eval_manifest.jsonl')
    parser.add_argument('--robotwin-root', type=Path, default=ROOT/'RoboTwin')
    parser.add_argument('--env-prefix', type=Path, default=ROOT/'.robotwin-env')
    parser.add_argument('--tasks-config', type=Path,
                        default=ROOT/'configs/robotwin/eval_frozen.yaml')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'eval_results/robotwin/audit')
    parser.add_argument('--task-config', choices=('demo_clean', 'demo_randomized', 'both'),
                        default='both')
    parser.add_argument('--gpu-ids', default='0', help='Comma-separated GPU ids, e.g. 0,1.')
    parser.add_argument('--jobs-per-gpu', default='1',
                        help='Blocks audited at once on each GPU: one value for all GPUs, '
                             'or one per entry of --gpu-ids (e.g. 4,2).')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker-task', help=argparse.SUPPRESS)
    parser.add_argument('--worker-setting', help=argparse.SUPPRESS)
    parser.add_argument('--worker-report', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    worker_values = (args.worker_task, args.worker_setting, args.worker_report)
    if any(worker_values):
        if not all(worker_values):
            parser.error('worker mode requires task, setting, and report')
        audit_block(args)
    else:
        schedule(args)


if __name__ == '__main__':
    main()
