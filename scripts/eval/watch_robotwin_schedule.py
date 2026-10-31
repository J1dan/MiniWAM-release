#!/usr/bin/env python3
"""Publish an atomic live report for one official RoboTwin multitask schedule."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time

ROOT = Path(__file__).resolve().parents[2]

ERROR =re.compile(r'\[(?:MiniWAM\]\[)?ERROR\]|Traceback \(most recent call last\):')
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
PROGRESS_RATE = re.compile(r'(?<!Final )(?:batch )?Success rate:\s*(\d+)\s*/\s*(\d+)', re.I)
FINAL_RATE = re.compile(r'Final(?: batch)? success rate:\s*(\d+)\s*/\s*(\d+)', re.I)
STATS_SHA256 = re.compile(r'\[MiniWAM\] dataset_stats_sha256=([0-9a-f]{64})')
PROVENANCE_KEYS = ('checkpoint_sha256', 'dataset_stats_sha256',
                   'eval_manifest_sha256', 'text_cache_sha256',
                   'robotwin_commit', 'xpolicylab_commit')


def checkpoint_label(checkpoint, digest, namespace=None):
    # The pinned checkpoint is a folder: outputs/<run>/step_<N> or checkpoints/<name>.
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent
    if run_dir.name == 'checkpoints':
        run_dir = run_dir.parent
    label = re.sub(r'[^A-Za-z0-9_.-]', '_',
                   f'{run_dir.name}__{checkpoint.name}')
    label += '__' + digest[:12]
    if namespace:
        label += '__' + re.sub(r'[^A-Za-z0-9_.-]', '_', str(namespace))
    return label


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


@functools.lru_cache(maxsize=None)
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def frozen_ledger_root(plan, checkpoint_sha):
    """Return the directory the MiniWAM adapter writes this schedule's ledgers to.

    Without an explicit case_output, eval.sh keys the default by checkpoint and by
    the frozen manifest's SHA256:
    eval_results/robotwin/frozen_case_ledgers/<checkpoint>/<manifest>[/renderer-*].
    """
    if plan.get('case_output'):
        return Path(plan['case_output']).expanduser().resolve()
    miniwam = plan.get('miniwam') or {}
    if not miniwam.get('manifest'):
        raise ValueError('A frozen schedule plan needs case_output or miniwam.manifest.')
    manifest = Path(miniwam['manifest']).expanduser().resolve()
    root = ROOT/'eval_results/robotwin/frozen_case_ledgers'/checkpoint_sha/file_sha256(str(manifest))
    renderer_profile = miniwam.get('renderer_compatibility', {}).get('profile', 'upstream')
    if renderer_profile != 'upstream':
        root = root/f'renderer-{renderer_profile}'
    return root.resolve()


def read_ledger(path, task, condition, target, checkpoint_sha):
    if not path.is_file():
        return []
    records, keys, indices = [], set(), set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Corrupt frozen ledger {path}:{line_number}') from exc
        if (record.get('task_name') != task or record.get('task_config') != condition or
                type(record.get('episode_index')) is not int or
                type(record.get('accepted_seed')) is not int or
                type(record.get('success')) is not bool):
            raise ValueError(f'Invalid frozen ledger record {path}:{line_number}')
        provenance = record.get('provenance', {})
        if (set(provenance) != set(PROVENANCE_KEYS) or
                any(not isinstance(provenance[key], str) or not provenance[key]
                    for key in PROVENANCE_KEYS)):
            raise ValueError(f'Incomplete frozen provenance in {path}:{line_number}')
        if provenance['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError(f'Checkpoint mismatch in frozen ledger {path}:{line_number}')
        key = (record['task_name'], record['task_config'], record['accepted_seed'])
        if key in keys:
            raise ValueError(f'Duplicate frozen case in {path}: {key}')
        if record['episode_index'] in indices:
            raise ValueError(f'Duplicate frozen episode index in {path}: {record["episode_index"]}')
        keys.add(key)
        indices.add(record['episode_index'])
        if record['episode_index'] < target:
            records.append(record)
    if len(records) > target:
        raise ValueError(f'Frozen ledger exceeds target for {task}/{condition}')
    return records


def refresh(schedule, output):
    plan = read_json(schedule/'schedule_plan.json')
    if not plan:
        raise FileNotFoundError(schedule/'schedule_plan.json')
    pinned = read_json(schedule/'miniwam_checkpoint.json')
    scheduler = read_json(schedule/'summary.json')
    scheduler_statuses = {job['job_id']: job.get('status') for job in scheduler.get('jobs', [])}
    now = time.time()
    protocol = plan.get('miniwam', {}).get('evaluation_protocol', 'frozen')
    if protocol not in {'frozen', 'official'}:
        raise ValueError(f'Unknown evaluation protocol in schedule plan: {protocol!r}')
    namespace = plan.get('result_namespace')
    base = dict(schema_version=3, schedule=str(schedule), schedule_id=schedule.name,
                checkpoint_request=plan['checkpoint_request'], checkpoint=None,
                checkpoint_sha256=None, status='initializing', timestamp=now,
                evaluation_protocol=protocol, result_namespace=namespace,
                renderer_compatibility=plan.get('miniwam', {}).get(
                    'renderer_compatibility', {'profile': 'upstream'}
                ),
                robotwin_commit=plan.get('robotwin_commit'),
                xpolicylab_commit=plan.get('xpolicylab_commit'),
                updated_at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                total_time_seconds=max(0, now-plan['created_at_epoch']),
                settings=dict(gpu_capacity=plan['gpu_capacity'],
                              parallel_workers=plan['parallel_workers'],
                              conditions=plan['conditions']),
                progress={}, overall_metrics={}, condition_metrics={}, results=[])
    destinations = [schedule/'report.json']
    if not pinned:
        base['progress'] = dict(total_tasks=len(plan['jobs']), completed_tasks=0,
            running_tasks=0, pending_tasks=len(plan['jobs']), failed_tasks=0,
            skipped_tasks=0, remaining_tasks=len(plan['jobs']), target_episodes=sum(
                int(j['overrides'].get('test_num', 100)) for j in plan['jobs']),
            completed_episodes=0, remaining_episodes=sum(
                int(j['overrides'].get('test_num', 100)) for j in plan['jobs']))
        for destination in destinations:
            atomic_json(destination, base)
        return base

    checkpoint = Path(pinned['path']).resolve()
    digest = pinned['sha256']
    base.update(checkpoint=str(checkpoint), checkpoint_sha256=digest)
    ledger_root = frozen_ledger_root(plan, digest) if protocol == 'frozen' else None
    rows, evidence = [], []
    for job in plan['jobs']:
        task = job['task']
        condition = job['overrides'].get('task_config', 'demo_clean')
        target = int(job['overrides'].get('test_num', 100))
        log = schedule/'logs'/f'{job["job_id"]}.log'
        text = log.read_text(errors='replace') if log.is_file() else ''
        plain = ANSI.sub('', text)
        scheduler_status = scheduler_statuses.get(job['job_id'])
        if protocol == 'frozen':
            records = read_ledger(ledger_root/f'{task}__{condition}.jsonl',
                                  task, condition, target, digest)
            evidence.extend(records)
            finished_episodes = len(records)
            successes = sum(record['success'] for record in records)
        else:
            final_matches = FINAL_RATE.findall(plain)
            progress_matches = PROGRESS_RATE.findall(plain)
            matches = final_matches or progress_matches
            successes, finished_episodes = ((int(value) for value in matches[-1])
                                             if matches else (0, 0))
            if finished_episodes > target or successes > finished_episodes:
                raise ValueError(f'Invalid official progress in {log}')
        if finished_episodes == target and ('[MiniWAM] task_completed=true' in plain or
                                             scheduler_status == 'success'):
            status = 'completed'
        elif scheduler_status in {'failed', 'skipped'}:
            status = scheduler_status
        elif scheduler and ERROR.search(text):
            status = 'failed'
        elif log.is_file() and not scheduler:
            status = 'running'
        elif log.is_file():
            status = 'interrupted'
        else:
            status = 'pending'
        rows.append(dict(task=task, condition=condition, seed=job['seed'], status=status,
            target_episodes=target, finished_episodes=finished_episodes, success_count=successes,
            failed_episode_count=finished_episodes-successes,
            success_rate=100*successes/finished_episodes if finished_episodes else None,
            remaining_episodes=target-finished_episodes, log_path=str(log) if log.is_file() else None,
            artifact_pointer=str((schedule/'jobs'/f'{job["job_id"]}.artifacts.json'))))
    counts = Counter(row['status'] for row in rows)
    finished = sum(row['finished_episodes'] for row in rows)
    successes = sum(row['success_count'] for row in rows)
    target = sum(row['target_episodes'] for row in rows)
    measured = [row for row in rows if row['finished_episodes']]
    provenances = {json.dumps(record['provenance'], sort_keys=True) for record in evidence}
    if len(provenances) > 1:
        raise ValueError('Selected frozen ledgers contain mixed provenance.')
    provenance = json.loads(next(iter(provenances))) if provenances else {}
    if protocol == 'official':
        statistics = set()
        for row in rows:
            if row['log_path']:
                statistics.update(STATS_SHA256.findall(
                    ANSI.sub('', Path(row['log_path']).read_text(errors='replace'))
                ))
        if len(statistics) > 1:
            raise ValueError('Official jobs resolved different dataset statistics.')
        provenance['dataset_stats_sha256'] = next(iter(statistics), None)
    complete = all(row['status'] == 'completed' for row in rows)
    if complete:
        status = 'complete'
    elif scheduler:
        status = 'incomplete'
    elif counts['running']:
        status = 'running'
    else:
        status = 'initializing'
    base.update(status=status, results=rows,
        dataset_stats_sha256=provenance.get('dataset_stats_sha256'),
        eval_manifest_sha256=provenance.get('eval_manifest_sha256'),
        text_cache_sha256=provenance.get('text_cache_sha256'),
        robotwin_commit=provenance.get('robotwin_commit', base.get('robotwin_commit')),
        xpolicylab_commit=provenance.get('xpolicylab_commit', base.get('xpolicylab_commit')),
        progress=dict(total_tasks=len(rows), completed_tasks=counts['completed'],
            running_tasks=counts['running'], pending_tasks=counts['pending'],
            failed_tasks=counts['failed'], skipped_tasks=counts['skipped'],
            interrupted_tasks=counts['interrupted'],
            remaining_tasks=len(rows)-counts['completed'], target_episodes=target,
            completed_episodes=finished, remaining_episodes=target-finished),
        overall_metrics=dict(average_success_rate=(sum(row['success_rate'] for row in measured)/len(measured)
                                                   if measured else None),
            total_tasks=len(rows), tasks_with_results=len(measured), total_episodes=finished,
            total_successes=successes, total_failed_episodes=finished-successes,
            execution_failures=counts['failed'], is_partial=not complete),
        condition_metrics={condition: dict(
            total_tasks=sum(row['condition'] == condition for row in rows),
            completed_tasks=sum(row['condition'] == condition and row['status'] == 'completed' for row in rows),
            total_episodes=sum(row['finished_episodes'] for row in rows if row['condition'] == condition),
            total_successes=sum(row['success_count'] for row in rows if row['condition'] == condition))
            for condition in plan['conditions']})
    destinations.append(output/checkpoint_label(checkpoint, digest, namespace)/'report.json')
    for destination in destinations:
        atomic_json(destination, base)
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'eval_results/robotwin/reports')
    parser.add_argument('--scheduler-pid', type=int)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=5.0)
    args = parser.parse_args()
    if args.watch:
        # A terminal Ctrl-C reaches the whole process group. The scheduler then stops
        # its jobs and writes summary.json, so keep running to publish the final report;
        # the loop ends when that file appears or the scheduler has exited.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    schedule = args.schedule_dir.resolve()
    lock_path = schedule/'report.lock'
    while True:
        # Hold the advisory lock only while publishing. A long-lived watcher
        # must not prevent a one-shot recovery refresh from repairing a stale
        # report after an interrupted evaluator.
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            report = refresh(schedule, args.output_dir.resolve())
        if not args.watch or (schedule/'summary.json').is_file():
            break
        if args.scheduler_pid and not Path(f'/proc/{args.scheduler_pid}').exists():
            break
        time.sleep(max(1.0, args.interval))
    print(f'Updated {schedule/"report.json"}: {report["status"]}')


if __name__ == '__main__':
    main()
