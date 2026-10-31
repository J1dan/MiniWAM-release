import hashlib
import json
from pathlib import Path

import pytest

import scripts.eval.watch_robotwin_schedule as watcher
from scripts.eval.watch_robotwin_schedule import refresh


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def record(task, condition, index, success, checkpoint='a' * 64):
    return dict(task_name=task, task_config=condition, accepted_seed=100000 + index,
                episode_index=index, success=success,
                provenance=dict(checkpoint_sha256=checkpoint,
                    dataset_stats_sha256='s'*64, eval_manifest_sha256='m'*64,
                    text_cache_sha256='t'*64, robotwin_commit='robotwin',
                    xpolicylab_commit='xpolicylab'))


def test_schedule_report_tracks_exact_ledgers_and_finishes(tmp_path):
    schedule = tmp_path/'schedule'; (schedule/'logs').mkdir(parents=True)
    (schedule/'jobs').mkdir()
    checkpoint = tmp_path/'outputs/run/step_000100'
    plan = dict(schema_version=1, created_at_epoch=1, checkpoint_request=str(checkpoint),
                gpu_capacity={'0': 2}, parallel_workers=2, conditions=['demo_clean'],
                case_output=str(tmp_path/'ledgers'), jobs=[
                    dict(job_id='000_task_a_seed0', task='task_a', seed=0,
                         overrides=dict(task_config='demo_clean', test_num=2)),
                    dict(job_id='001_task_b_seed0', task='task_b', seed=0,
                         overrides=dict(task_config='demo_clean', test_num=1)),
                ])
    write_json(schedule/'schedule_plan.json', plan)
    write_json(schedule/'miniwam_checkpoint.json', dict(path=str(checkpoint), sha256='a'*64))
    (schedule/'logs/000_task_a_seed0.log').write_text('running\n')
    (tmp_path/'ledgers').mkdir()
    (tmp_path/'ledgers/task_a__demo_clean.jsonl').write_text(
        json.dumps(record('task_a', 'demo_clean', 0, True)) + '\n')
    report = refresh(schedule, tmp_path/'view')
    assert report['status'] == 'running'
    assert report['progress']['target_episodes'] == 3
    assert report['progress']['completed_episodes'] == 1
    assert report['progress']['remaining_episodes'] == 2
    assert report['overall_metrics']['total_successes'] == 1
    assert report['dataset_stats_sha256'] == 's'*64
    assert (tmp_path/'view/run__step_000100__aaaaaaaaaaaa/report.json').is_file()

    (tmp_path/'ledgers/task_a__demo_clean.jsonl').write_text('\n'.join(json.dumps(value) for value in (
        record('task_a', 'demo_clean', 0, True), record('task_a', 'demo_clean', 1, False))) + '\n')
    (tmp_path/'ledgers/task_b__demo_clean.jsonl').write_text(
        json.dumps(record('task_b', 'demo_clean', 0, True)) + '\n')
    write_json(schedule/'summary.json', dict(jobs=[
        dict(job_id='000_task_a_seed0', status='success'),
        dict(job_id='001_task_b_seed0', status='success')]))
    report = refresh(schedule, tmp_path/'view')
    assert report['status'] == 'complete'
    assert report['progress']['completed_tasks'] == 2
    assert report['overall_metrics']['total_episodes'] == 3
    assert report['overall_metrics']['average_success_rate'] == 75.0


def test_schedule_report_rejects_wrong_checkpoint_ledger(tmp_path):
    schedule = tmp_path/'schedule'; (schedule/'logs').mkdir(parents=True)
    write_json(schedule/'schedule_plan.json', dict(schema_version=1, created_at_epoch=1,
        checkpoint_request='model.pt', gpu_capacity={'0': 1}, parallel_workers=1,
        conditions=['demo_clean'], case_output=str(tmp_path/'ledgers'), jobs=[
            dict(job_id='000_task_seed0', task='task', seed=0,
                 overrides=dict(task_config='demo_clean', test_num=1))]))
    write_json(schedule/'miniwam_checkpoint.json', dict(path=str(tmp_path/'model.pt'), sha256='a'*64))
    (tmp_path/'ledgers').mkdir()
    (tmp_path/'ledgers/task__demo_clean.jsonl').write_text(
        json.dumps(record('task', 'demo_clean', 0, True, checkpoint='b'*64)) + '\n')
    with pytest.raises(ValueError, match='Checkpoint mismatch'):
        refresh(schedule, tmp_path/'view')


@pytest.mark.parametrize('profile', ['upstream', 'oidn_2_3_3_blackwell'])
def test_schedule_report_reads_adapter_default_ledgers(tmp_path, monkeypatch, profile):
    # Without case_output, eval.sh writes the ledgers under
    # eval_results/robotwin/frozen_case_ledgers/<checkpoint>/<manifest SHA256>[/renderer-<profile>].
    monkeypatch.setattr(watcher, 'ROOT', tmp_path)
    manifest = tmp_path/'bundle/robotwin_eval_manifest.jsonl'
    manifest.parent.mkdir()
    manifest.write_text('{"task_name": "task"}\n')
    schedule = tmp_path/'schedule'; (schedule/'logs').mkdir(parents=True)
    write_json(schedule/'schedule_plan.json', dict(schema_version=1, created_at_epoch=1,
        checkpoint_request='model.pt', gpu_capacity={'0': 1}, parallel_workers=1,
        conditions=['demo_clean'], case_output=None,
        miniwam=dict(evaluation_protocol='frozen', manifest=str(manifest),
                     renderer_compatibility=dict(profile=profile)), jobs=[
            dict(job_id='000_task_seed0', task='task', seed=0,
                 overrides=dict(task_config='demo_clean', test_num=1))]))
    write_json(schedule/'miniwam_checkpoint.json', dict(path=str(tmp_path/'model.pt'), sha256='a'*64))
    (schedule/'logs/000_task_seed0.log').write_text('[MiniWAM] task_completed=true\n')
    ledgers = (tmp_path/'eval_results/robotwin/frozen_case_ledgers'/('a'*64)/
               hashlib.sha256(manifest.read_bytes()).hexdigest())
    if profile != 'upstream':
        ledgers = ledgers/f'renderer-{profile}'
    ledgers.mkdir(parents=True)
    (ledgers/'task__demo_clean.jsonl').write_text(
        json.dumps(record('task', 'demo_clean', 0, True)) + '\n')
    report = refresh(schedule, tmp_path/'view')
    assert report['progress']['completed_tasks'] == 1
    assert report['progress']['completed_episodes'] == 1
    assert report['overall_metrics']['total_successes'] == 1


def test_schedule_report_tracks_official_log_without_frozen_ledger(tmp_path):
    schedule = tmp_path/'schedule'; (schedule/'logs').mkdir(parents=True)
    checkpoint = tmp_path/'outputs/run/step_000100'
    write_json(schedule/'schedule_plan.json', dict(schema_version=2, created_at_epoch=1,
        checkpoint_request=str(checkpoint), gpu_capacity={'0': 1}, parallel_workers=1,
        conditions=['demo_clean'], miniwam={'evaluation_protocol': 'official'},
        result_namespace='official-seed0', case_output=None, jobs=[
            dict(job_id='000_task_seed0', task='task', seed=0,
                 overrides=dict(task_config='demo_clean', test_num=3, expert_check=True))]))
    write_json(schedule/'miniwam_checkpoint.json', dict(path=str(checkpoint), sha256='a'*64))
    (schedule/'logs/000_task_seed0.log').write_text(
        '[MiniWAM] evaluation_protocol=official\n'
        '[MiniWAM] dataset_stats_sha256=' + 'b'*64 + '\n'
        'Success rate: 1/2 => 50.0%\n')
    report = refresh(schedule, tmp_path/'view')
    assert report['evaluation_protocol'] == 'official'
    assert report['progress']['completed_episodes'] == 2
    assert report['overall_metrics']['total_successes'] == 1
    assert report['dataset_stats_sha256'] == 'b'*64
    assert (tmp_path/'view/run__step_000100__aaaaaaaaaaaa__official-seed0/report.json').is_file()

    (schedule/'logs/000_task_seed0.log').write_text(
        '[MiniWAM] evaluation_protocol=official\n'
        '[MiniWAM] dataset_stats_sha256=' + 'b'*64 + '\n'
        'Final success rate: 2/3 = 66.7%\n[MiniWAM] task_completed=true\n')
    write_json(schedule/'summary.json', dict(jobs=[dict(job_id='000_task_seed0', status='success')]))
    report = refresh(schedule, tmp_path/'view')
    assert report['status'] == 'complete'
    assert report['progress']['completed_tasks'] == 1
    assert report['overall_metrics']['total_episodes'] == 3


@pytest.mark.parametrize('checkpoint, label', [
    ('outputs/robotwin_stage2/step_042880', 'robotwin_stage2__step_042880__aaaaaaaaaaaa'),
    ('MiniWAM/checkpoints/robotwin', 'MiniWAM__robotwin__aaaaaaaaaaaa'),
])
def test_checkpoint_report_folder_is_named_after_the_pinned_folder(tmp_path, checkpoint, label):
    # The adapter pins the checkpoint folder itself, not a file inside it.
    assert watcher.checkpoint_label(tmp_path/checkpoint, 'a'*64) == label


def test_adapter_scheduler_and_watcher_share_the_results_layout():
    adapter = Path('integrations/xpolicylab/policy/MiniWAM/eval.sh').read_text()
    scheduler_patch = Path('integrations/robotwin/robotwin.patch').read_text()
    source = Path(watcher.__file__).read_text()
    assert '"${project_root}/eval_results/robotwin/frozen_case_ledgers/' in adapter
    assert "ROOT/'eval_results/robotwin/frozen_case_ledgers'" in source
    assert 'str(miniwam_root / "eval_results" / "robotwin" / "reports")' in scheduler_patch
    assert "default=ROOT/'eval_results/robotwin/reports'" in source
