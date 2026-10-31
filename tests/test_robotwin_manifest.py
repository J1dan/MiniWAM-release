import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import numpy as np
import yaml

from miniwam.robotwin_manifest import validate_cases, sha256, content_hash, portable_config, atomic_json


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def cases():
    return [dict(task_name=f'task{i}', task_config=setting, accepted_seed=100000+j,
                 instruction_type='seen', episode_index=j, instruction='  Exact Phrase.  ')
            for i in range(50) for setting in ('demo_clean', 'demo_randomized') for j in range(100)]


def test_case_cardinality_and_uniqueness():
    rows = cases()
    tasks = [f'task{i}' for i in range(50)]
    validate_cases(rows, tasks)
    with pytest.raises(ValueError, match='Duplicate'):
        validate_cases(rows + [rows[0]], tasks)
    with pytest.raises(ValueError, match='exactly'):
        validate_cases(rows[:-1], tasks)


def test_portable_config_and_strict_scene_hash():
    assert content_hash(portable_config({'asset': '/server1/RoboTwin/assets/a'}, '/server1/RoboTwin')) == \
           content_hash(portable_config({'asset': '/server2/RoboTwin/assets/a'}, '/server2/RoboTwin'))
    assert content_hash({'pose': [0, 1, 2]}) != content_hash({'pose': [0, 1, 2.000001]})
    with pytest.raises(ValueError):
        content_hash({'pose': float('nan')})
    with pytest.raises(TypeError):
        content_hash({'object': object()})


def test_full_scene_audit_diff():
    audit_tool = module(
        'frozen_manifest_audit',
        ROOT/'scripts/eval/audit_robotwin_frozen_manifest.py',
    )
    assert audit_tool.difference_paths(
        {'models': {'object': {'extents': [1, 2, 3]}}},
        {'models': {'object': {'extents': [1, 9]}}},
    ) == [
        'models.object.extents.length',
        'models.object.extents[1]',
    ]


def test_audit_hands_every_block_to_the_listed_gpus_once(tmp_path, monkeypatch, capsys):
    audit_tool = module('frozen_manifest_audit_schedule',
                        ROOT/'scripts/eval/audit_robotwin_frozen_manifest.py')
    manifest = SimpleNamespace(load_manifest=lambda path: (None, {'eval_manifest_sha256': 'sha'}))
    monkeypatch.setattr(audit_tool, 'load_module', lambda name, path: manifest)
    tasks = tmp_path/'tasks.yaml'
    tasks.write_text(yaml.safe_dump({'tasks': [f'task{i}' for i in range(50)]}))
    args = SimpleNamespace(manifest=tmp_path/'manifest.jsonl', tasks_config=tasks,
                           task_config='both', output_dir=tmp_path/'audit', gpu_ids='0,1',
                           jobs_per_gpu='3,1', dry_run=True, robotwin_root=tmp_path,
                           env_prefix=tmp_path/'env')
    audit_tool.schedule(args)
    planned = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    blocks = sorted((entry['command'][entry['command'].index('--worker-task') + 1],
                     entry['command'][entry['command'].index('--worker-setting') + 1])
                    for entry in planned)
    assert blocks == sorted((f'task{i}', setting) for i in range(50)
                            for setting in ('demo_clean', 'demo_randomized'))
    assert {entry['gpu'] for entry in planned} <= {0, 1}
    with pytest.raises(ValueError, match='one per entry of --gpu-ids'):
        audit_tool.schedule(SimpleNamespace(**{**vars(args), 'jobs_per_gpu': '1,2,3'}))


@pytest.mark.parametrize('mismatches', [0, 1])
def test_audit_passes_only_when_every_scene_is_rebuilt_exactly(tmp_path, monkeypatch, capsys,
                                                               mismatches):
    audit_tool = module('frozen_manifest_audit_verdict',
                        ROOT/'scripts/eval/audit_robotwin_frozen_manifest.py')
    manifest = SimpleNamespace(load_manifest=lambda path: (None, {'eval_manifest_sha256': 'sha'}),
                               atomic_json=atomic_json)
    monkeypatch.setattr(audit_tool, 'load_module', lambda name, path: manifest)
    tasks = [f'task{i}' for i in range(50)]
    (tmp_path/'tasks.yaml').write_text(yaml.safe_dump({'tasks': tasks}))
    blocks = tmp_path/'audit/blocks'
    blocks.mkdir(parents=True)
    audit_hash, scene_hash, hook_hash = audit_tool.protocol_hashes()
    for task in tasks:
        for setting in ('demo_clean', 'demo_randomized'):
            # Every block is already audited, so no worker runs; one may hold a mismatch.
            bad = mismatches if (task, setting) == ('task0', 'demo_clean') else 0
            (blocks/f'{task}__{setting}.json').write_text(json.dumps(dict(
                complete=True, manifest_sha256='sha', audit_tool_sha256=audit_hash,
                scene_protocol_sha256=scene_hash, replay_hook_sha256=hook_hash,
                replay_state_sha256=audit_tool.replay_state_hash(), task_name=task,
                task_config=setting, checked_cases=100, setup_error_count=0,
                exact_cases=100 - bad, scene_mismatch_count=bad, cases=[{}] * 100)))
    args = SimpleNamespace(manifest=tmp_path/'manifest.jsonl', tasks_config=tmp_path/'tasks.yaml',
                           task_config='both', output_dir=tmp_path/'audit', gpu_ids='0',
                           jobs_per_gpu='1', dry_run=False, robotwin_root=tmp_path,
                           env_prefix=tmp_path/'env')
    if mismatches:
        with pytest.raises(SystemExit) as exit_info:
            audit_tool.schedule(args)
        assert exit_info.value.code == 1
        assert 'FAIL: 9999/10000 cases rebuilt exactly; 1 blocks' in capsys.readouterr().err
    else:
        audit_tool.schedule(args)
        assert 'PASS: all 10000 cases rebuilt exactly' in capsys.readouterr().out
    audit = json.loads((tmp_path/'audit/audit.json').read_text())
    assert audit['exact_cases'] == 10000 - mismatches
    assert audit['scene_mismatch_cases'] == mismatches


def test_release_model_choice_replay_preserves_rng_and_forces_manifest_variant(tmp_path):
    hooks = module('release_choice_replay', ROOT/'integrations/xpolicylab/policy/MiniWAM/compat/frozen_cases.py')
    model_root = tmp_path/'assets'/'objects'/'050_bell'
    model_root.mkdir(parents=True)
    (model_root/'model_data0.json').write_text(json.dumps({'scale': [0.04, 0.04, 0.04]}))
    expected = {'scale': [0.05, 0.05, 0.05]}
    (model_root/'model_data1.json').write_text(json.dumps(expected))
    case = dict(task_name='place_object_scale', scene_snapshot=dict(
        identities={'selected_modelname': '050_bell'}, models={'object': expected}))

    np.random.seed(7)
    overrides = hooks.release_model_choice_overrides(case, tmp_path)
    assert [(target, name, key) for _, target, name, key in overrides] == [
        (1, '050_bell', 'object')
    ]
    replay = hooks.ReleaseModelChoiceReplay(case, tmp_path)
    replay.install()
    try:
        assert int(np.random.choice([0, 1])) == 1
    finally:
        replay.restore()
    after_forced = np.random.random()

    np.random.seed(7)
    np.random.choice([0, 1])
    after_official = np.random.random()
    assert after_forced == after_official


def test_release_scene_compatibility_is_limited_to_stale_stack_planner_alias():
    hooks = module('release_scene_compat', ROOT/'integrations/xpolicylab/policy/MiniWAM/compat/frozen_cases.py')
    expected = {'actors': [{'pose': [1, 2, 3]}], 'models': {'block1': {'scale': 1},
                                                            'last_actor': {'scale': 1}}}
    reconstructed = {'actors': [{'pose': [1, 2, 3]}], 'models': {'block1': {'scale': 1}}}
    case = dict(task_name='stack_blocks_two', scene_snapshot=expected,
                scene_sha256=content_hash(expected))
    mode, digest = hooks.validate_release_scene(case, reconstructed)
    assert mode == 'release_v1_ignore_stale_planner_last_actor'
    assert digest == content_hash(reconstructed)

    reconstructed['actors'][0]['pose'][0] += 1e-9
    with pytest.raises(ValueError, match='differs'):
        hooks.validate_release_scene(case, reconstructed)

    case['task_name'] = 'another_task'
    reconstructed['actors'][0]['pose'][0] = 1
    with pytest.raises(ValueError, match='differs'):
        hooks.validate_release_scene(case, reconstructed)


def test_bundle_relocation_and_corruption(tmp_path):
    package = module('package', ROOT/'scripts/data/package_robotwin_eval_cache.py')
    with pytest.raises(ValueError, match='10000/10000'):
        package.validate_v2_certification({
            'schema_version': 2,
            'release_status': 'candidate_requires_full_exact_replay_audit',
        })
    package.validate_v2_certification({
        'schema_version': 2,
        'release_status': 'certified_exact_local_replay',
        'strict_replay_validation': {
            'exact_cases': 10000, 'expected_cases': 10000,
            'scene_mismatch_cases': 0, 'setup_error_cases': 0,
        },
    })
    bundle = tmp_path/'original'
    cache = bundle/'text_cache'; cache.mkdir(parents=True)
    manifest = bundle/'robotwin_eval_manifest.jsonl'
    rows = cases()
    for row in rows:
        row.update(scene_snapshot={'pose': [0, 1, 2]}, scene_sha256=content_hash({'pose': [0, 1, 2]}))
    manifest.write_text(''.join(json.dumps(c)+'\n' for c in rows))
    atomic_json(manifest.with_suffix('.provenance.json'), dict(tasks=[f'task{i}' for i in range(50)],
        eval_manifest_sha256=sha256(manifest), prism_commit='a'*40,
        prism_source_hashes={'source.py': 'b'*64}, instruction_validation={'valid_instructions':10000}))
    payload = cache/('a'*64+'.pt'); payload.write_bytes(b'synthetic embedding fixture')
    index = cache/'frozen_index.json'
    atomic_json(index, dict(eval_manifest_sha256=sha256(manifest), tokenizer_files={}, contract={},
        entries=[dict(instruction=rows[0]['instruction'], file=payload.name, embedding_sha256=sha256(payload))]))
    import hashlib
    atomic_json(cache/'validation.json', dict(total=1, exact_matches=1, contract={},
        instructions_sha256=hashlib.sha256(json.dumps([rows[0]['instruction']]).encode()).hexdigest()))
    atomic_json(index.with_suffix('.sha256.json'), dict(text_cache_sha256=sha256(index)))
    atomic_json(bundle/'bundle.json', dict(files={str(p.relative_to(bundle)): sha256(p)
        for p in bundle.rglob('*') if p.is_file()}))
    atomic_json(bundle/'bundle.sha256.json', dict(bundle_sha256=sha256(bundle/'bundle.json')))
    relocated = tmp_path/'another_server'
    bundle.rename(relocated)
    package.verify(relocated)
    extra = relocated/'source.py'; extra.write_text('should never upload')
    with pytest.raises(ValueError, match='Unexpected files'):
        package.verify(relocated)
    extra.unlink()
    (relocated/'text_cache'/payload.name).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        package.verify(relocated)


def test_replay_subset_then_rest_and_fail_closed(tmp_path, monkeypatch):
    hooks = module('frozen_cases_test', ROOT/'integrations/xpolicylab/policy/MiniWAM/compat/frozen_cases.py')
    rows = cases()
    provenance = dict(eval_manifest_sha256='manifest', sources=dict(robotwin_commit='r', xpolicylab_commit='x'))
    provenance['generation_script_sha256'] = sorted(hooks.RELEASED_CAPTURE_HOOK_SHA256)[0]
    provenance['scene_protocol_sha256'] = sha256(ROOT/'miniwam/robotwin_manifest.py')
    snapshot_episode_info = []
    def snapshot(_env, episode_info):
        snapshot_episode_info.append(episode_info)
        return {'object': 'fan', 'pose': [0, 1, 2]}
    monkeypatch.setattr(hooks.protocol, 'scene_snapshot', snapshot)
    monkeypatch.setattr(hooks, 'load_manifest', lambda *a: (rows, provenance))
    checkpoint = tmp_path/'checkpoint'; checkpoint.mkdir()
    (checkpoint/'model.safetensors').write_bytes(b'weights')
    (checkpoint/'statistics.json').write_text('{}')
    index = tmp_path/'index.json'
    index.write_text(json.dumps(dict(eval_manifest_sha256='manifest', entries=[])))
    index.with_suffix('.sha256.json').write_text(json.dumps(dict(text_cache_sha256=sha256(index))))
    for name, value in dict(MINIWAM_CASE_MODE='replay', MINIWAM_CASE_OUTPUT=str(tmp_path/'ledger'),
        MINIWAM_EVAL_MANIFEST='manifest', MINIWAM_FROZEN_TEXT_INDEX=str(index),
        MINIWAM_CHECKPOINT_PATH=str(checkpoint)).items():
        monkeypatch.setenv(name, value)
    args = dict(task_name='task0', task_config='demo_clean')
    usr = dict(root_dir=str(tmp_path))
    setup_hash = content_hash(dict(**args, action_type='joint', frequency=30))
    for row in rows:
        row['setup_sha256'] = setup_hash
    first = hooks.FrozenCases(args, usr, tmp_path/'attempt1', 'seen', 20)
    assert usr['expert_check'] is False
    assert first.replay_metadata['replay_expert_planner_executed'] is False
    import numpy as np
    import random
    state = np.random.get_state()
    for row in rows:
        row.update(setup_sha256=first.setup_hash,
            scene_snapshot={'episode_info': {'info': {'object': 'fan'}}},
            scene_sha256=hooks.protocol.content_hash({'object': 'fan', 'pose': [0, 1, 2]}),
            numpy_rng_state=[state[0], state[1].tolist(), *state[2:]],
            python_rng_state=json.loads(json.dumps(random.getstate())))
    assert first.count == 20
    for i in range(20):
        seed = first.seed(i, -1)
        assert first.instruction() == '  Exact Phrase.  '
        first.accepted(seed, i, first.instruction(), {'wrong': 'runtime placeholder'}, None)
        first.result(i % 2 == 0)
    assert snapshot_episode_info == [
        {'info': {'object': 'fan'}} for _ in range(20)
    ]
    assert first.finish() == 10  # Unsuccessful rollouts are completed cases too.
    with pytest.raises(ValueError, match='unvalidated'):
        first.result(True)
    first.lock.close()
    with pytest.raises(ValueError, match='setup differs'):
        hooks.FrozenCases(dict(**args, changed_setting=True), usr,
                          tmp_path/'changed_setup', 'seen', 20)
    second = hooks.FrozenCases(args, usr, tmp_path/'attempt2', 'seen', 100)
    assert second.count == 80
    assert second.seed(0, -1) == 100020
    with pytest.raises(ValueError, match='mismatch'):
        second.accepted(100021, 0, second.instruction(), {}, None)
    monkeypatch.setattr(hooks.protocol, 'scene_snapshot', lambda *args: {'object': 'other fan', 'pose': [0, 1, 2]})
    with pytest.raises(ValueError, match='scene differs'):
        second.accepted(100020, 0, second.instruction(), {}, None)
    with pytest.raises(ValueError, match='missing'):
        second.finish()
    second.lock.close()
    (checkpoint/'model.safetensors').write_bytes(b'other checkpoint')
    with pytest.raises(ValueError, match='provenance'):
        hooks.FrozenCases(args, usr, tmp_path/'attempt3', 'seen', 100)


def test_replay_ledger_atomic_commit_preserves_prior_results(tmp_path, monkeypatch):
    hooks = module('atomic_ledger', ROOT/'integrations/xpolicylab/policy/MiniWAM/compat/frozen_cases.py')
    path = tmp_path/'task.jsonl'
    hooks.write_replay_ledger(path, [{'success': True}])
    original = path.read_bytes()
    def interrupted_replace(*args):
        raise OSError('simulated interruption before commit')
    monkeypatch.setattr(hooks.os, 'replace', interrupted_replace)
    with pytest.raises(OSError, match='interruption'):
        hooks.write_replay_ledger(path, [{'success': True}, {'success': False}])
    assert path.read_bytes() == original


def test_eval_exit_trap_stops_server_without_masking_status():
    import subprocess
    source = (ROOT/'integrations/xpolicylab/policy/MiniWAM/eval.sh').read_text()
    assert source.count('trap stop_policy_server EXIT') == 1
    function = source[source.index('stop_policy_server() {'):source.index('trap stop_policy_server EXIT')]
    script = 'set -euo pipefail\n' + function + '''
SERVER_PID=123
kill() { echo cleanup; }
wait() { :; }
trap stop_policy_server EXIT
exit "$1"
'''
    for evaluation_code in ('0', '3'):
        result = subprocess.run(['bash', '-c', script, 'test', evaluation_code],
                                capture_output=True, text=True)
        assert result.returncode == int(evaluation_code)
        assert 'cleanup' in result.stdout
