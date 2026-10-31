import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.eval import robotwin_protocol
from scripts.eval.robotwin_protocol import PROTOCOL_ENVIRONMENT, resolve_protocol_config


def _write_frozen_assets(root):
    manifest = root/'manifest.jsonl'
    manifest.write_text('{"case": 1}\n')
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.with_suffix('.provenance.json').write_text(json.dumps({
        'eval_manifest_sha256': manifest_sha,
        'schema_version': 2, 'benchmark_protocol': 'official_split_v1',
        'release_status': 'certified_exact_local_replay',
        'strict_replay_validation': dict(exact_cases=10000, expected_cases=10000,
                                        scene_mismatch_cases=0, setup_error_cases=0),
    }))
    index = root/'frozen_index.json'
    index.write_text(json.dumps({'eval_manifest_sha256': manifest_sha}))
    index.with_suffix('.sha256.json').write_text(json.dumps({
        'text_cache_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
    }))
    return manifest, index


def test_protocol_config_resolves_two_disjoint_modes(tmp_path):
    manifest, index = _write_frozen_assets(tmp_path)
    frozen, frozen_env = resolve_protocol_config({
        'evaluation_protocol': 'frozen', 'manifest': str(manifest),
        'text_index': str(index), 'case_output': 'ledgers',
    }, tmp_path)
    assert frozen['result_namespace'].startswith('frozen-')
    assert frozen_env['MINIWAM_CASE_MODE'] == 'replay'
    assert frozen_env['MINIWAM_TEXT_CACHE_MISS'] == 'error'
    assert frozen_env['MINIWAM_CASE_OUTPUT'] == str((tmp_path/'ledgers').resolve())
    assert frozen['renderer_compatibility'] == {'profile': 'upstream'}
    assert frozen_env['MINIWAM_ROBOTWIN_RENDER_PROFILE'] == 'upstream'

    official, official_env = resolve_protocol_config({
        'evaluation_protocol': 'official', 'text_cache_dir': 'online-cache',
    }, tmp_path)
    assert official['result_namespace'] == 'official'
    assert official_env['MINIWAM_TEXT_CACHE_MISS'] == 'build'
    assert not {'MINIWAM_CASE_MODE', 'MINIWAM_EVAL_MANIFEST',
                'MINIWAM_FROZEN_TEXT_INDEX'} & set(official_env)
    assert 'MINIWAM_CASE_OUTPUT' in PROTOCOL_ENVIRONMENT
    assert 'MINIWAM_OIDN_ROOT' in PROTOCOL_ENVIRONMENT


def test_blackwell_oidn_profile_is_opt_in_hash_bound_and_namespaced(tmp_path, monkeypatch):
    manifest, index = _write_frozen_assets(tmp_path)
    oidn_root = tmp_path / 'oidn'
    library_dir = oidn_root / 'lib'
    library_dir.mkdir(parents=True)
    expected = {
        'libOpenImageDenoise_core.so.2.3.3':
            'eadb2437d6ac4a88e5746980495ba1091f7d14745e3000b82a9382093d22f351',
        'libOpenImageDenoise.so.2.3.3':
            '9ac9dcd18318cae879efb01c14cff72c2c4275428a15e5ab6e94c8ff8a43c4dc',
        'libOpenImageDenoise_device_cuda.so.2.3.3':
            '12b3ff87cfa4b6a90481b757f5a56ee69380a45e3dcaeadf8dd22cb73d4f714a',
    }
    for name in expected:
        (library_dir / name).write_bytes(b'test')
    real_sha256 = robotwin_protocol.sha256
    monkeypatch.setattr(robotwin_protocol, 'sha256', lambda path:
        expected[Path(path).name] if Path(path).name in expected else real_sha256(path))
    config, environment = resolve_protocol_config({
        'evaluation_protocol': 'frozen', 'manifest': str(manifest),
        'text_index': str(index), 'renderer_compatibility': {
            'profile': 'oidn_2_3_3_blackwell', 'oidn_root': str(oidn_root),
        },
    }, tmp_path)
    assert config['result_namespace'].endswith('-renderer-oidn_2_3_3_blackwell')
    assert config['renderer_compatibility']['library_sha256'] == expected
    assert environment['MINIWAM_ROBOTWIN_RENDER_PROFILE'] == 'oidn_2_3_3_blackwell'
    assert environment['MINIWAM_OIDN_ROOT'] == str(oidn_root.resolve())

    config, environment = resolve_protocol_config({
        'evaluation_protocol': 'frozen', 'manifest': str(manifest),
        'text_index': str(index), 'case_output': 'ledgers',
        'renderer_compatibility': {
            'profile': 'oidn_2_3_3_blackwell', 'oidn_root': str(oidn_root),
        },
    }, tmp_path)
    assert environment['MINIWAM_CASE_OUTPUT'].endswith(
        '/ledgers/renderer-oidn_2_3_3_blackwell'
    )


def test_protocol_config_rejects_cross_mode_fields_and_bad_seal(tmp_path):
    manifest, index = _write_frozen_assets(tmp_path)
    with pytest.raises(ValueError, match='Unsupported official'):
        resolve_protocol_config({'evaluation_protocol': 'official',
                                 'manifest': str(manifest)}, tmp_path)
    index.with_suffix('.sha256.json').write_text(json.dumps({'text_cache_sha256': 'bad'}))
    with pytest.raises(ValueError, match='text-index SHA256'):
        resolve_protocol_config({'evaluation_protocol': 'frozen',
                                 'manifest': str(manifest), 'text_index': str(index)}, tmp_path)


def test_eval_configs_are_explicit():
    frozen = yaml.safe_load(Path('configs/robotwin/eval_frozen.yaml').read_text())
    official = yaml.safe_load(Path('configs/robotwin/eval_official.yaml').read_text())
    assert frozen['miniwam']['evaluation_protocol'] == 'frozen'
    assert official['miniwam']['evaluation_protocol'] == 'official'
    assert frozen['tasks'] == official['tasks']
    assert len(frozen['tasks']) == 50


def test_frozen_rejects_uncertified_cache(tmp_path):
    manifest, index = _write_frozen_assets(tmp_path)
    path = manifest.with_suffix('.provenance.json')
    original = json.loads(path.read_text())
    config = dict(evaluation_protocol='frozen', manifest=str(manifest), text_index=str(index))
    candidate = dict(original, release_status='candidate_requires_full_exact_replay_audit')
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match='certification'):
        resolve_protocol_config(config, tmp_path)
    original.pop('benchmark_protocol')
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='certified official-split release'):
        resolve_protocol_config(config, tmp_path)


@pytest.mark.robotwin_install
@pytest.mark.skipif(not Path("RoboTwin/scripts/eval_policy_multitask.py").is_file(), reason="Run scripts/setup_robotwin.sh for simulator integration tests")
def test_shared_preflight_rejects_wrong_instruction_and_batch_seed_selection():
    from types import SimpleNamespace
    from scripts.eval.robotwin_protocol import validate_evaluation_jobs
    config = dict(evaluation_protocol='official')
    job = SimpleNamespace(task_name='adjust_bottle', seed=0, action_type='joint',
        env_cfg_type='aloha_agilex', overrides=dict(task_config='demo_randomized',
        test_num=100, expert_check=True, num_workers=1, eval_batch=False))
    report = validate_evaluation_jobs([job], config, Path('RoboTwin'))
    assert report['split_configs']['demo_randomized']['instruction_type'] == 'unseen'
    for field, value, message in [('instruction_type', 'seen', 'requires unseen'),
                                  ('num_workers', 2, 'one simulator'),
                                  ('eval_batch', True, 'one simulator'),
                                  ('expert_check', False, 'expert-check'),
                                  ('test_num', 101, '1..100'),
                                  ('frequency', 20, '30 Hz')]:
        wrong = SimpleNamespace(**{**vars(job), 'overrides': {**job.overrides, field: value}})
        with pytest.raises(ValueError, match=message):
            validate_evaluation_jobs([wrong], config, Path('RoboTwin'))


def _patch_upstream_evaluator(root):
    import subprocess
    subprocess.check_call(['git', 'init', '-q', str(root)])
    for name in ('eval_policy.sh', 'eval_policy_xpolicylab.py', 'eval_policy_multitask.py'):
        upstream = subprocess.check_output(['git', '-C', 'RoboTwin', 'show',
            f'96c1feab536306b50c26af200044fcdf126e8904:scripts/{name}'])
        (root/'scripts').mkdir(exist_ok=True)
        (root/'scripts'/name).write_bytes(upstream)
    patch = Path('integrations/robotwin/robotwin.patch').resolve()
    subprocess.check_call(['git', '-C', str(root), 'apply', str(patch)])


@pytest.mark.skipif(not Path('RoboTwin/.git').exists(), reason='Run scripts/setup_robotwin.sh first')
def test_robotwin_patch_reproduces_the_certified_evaluator(tmp_path):
    from scripts.eval.robotwin_protocol import OFFICIAL_CAPTURE_SHA256, sha256
    _patch_upstream_evaluator(tmp_path)
    assert sha256(tmp_path/'scripts/eval_policy_xpolicylab.py') == OFFICIAL_CAPTURE_SHA256
    scheduler = (tmp_path/'scripts/eval_policy_multitask.py').read_text()
    compile(scheduler, 'eval_policy_multitask.py', 'exec')
    assert 'settings["miniwam"] = config.get("miniwam")' in scheduler
    assert 'validate_evaluation_jobs(jobs, protocol, ROBOTWIN_ROOT)' in scheduler


@pytest.mark.skipif(not Path('RoboTwin/.git').exists(), reason='Run scripts/setup_robotwin.sh first')
def test_robotwin_launcher_takes_gpus_and_jobs_per_gpu_on_the_command_line(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace
    _patch_upstream_evaluator(tmp_path)
    spec = importlib.util.spec_from_file_location(
        'patched_robotwin_scheduler', tmp_path/'scripts/eval_policy_multitask.py')
    scheduler = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, scheduler)  # its dataclasses look the module up
    spec.loader.exec_module(scheduler)

    def capacity(config, gpu_ids=None, jobs_per_gpu=None):
        return scheduler.parse_gpu_capacity(
            config, SimpleNamespace(gpu_ids=gpu_ids, jobs_per_gpu=jobs_per_gpu))

    assert capacity({'gpu_ids': '0', 'jobs_per_gpu': 1}) == {'0': 1}
    assert capacity({'gpu_ids': '0', 'jobs_per_gpu': 1}, '0-1', '2,4') == {'0': 2, '1': 4}
    assert capacity({'gpu_ids': '2,3'}, jobs_per_gpu='3') == {'2': 3, '3': 3}
    assert capacity({'gpu_ids': [2, 3], 'jobs_per_gpu': [1, 3]}) == {'2': 1, '3': 3}
    assert scheduler.gpu_slots(capacity({'gpu_ids': '0,1', 'jobs_per_gpu': [1, 3]})) == \
        ['0', '1', '1', '1']
    for config, message in [({'gpu_ids': '0,1', 'jobs_per_gpu': '1,2,3'}, 'one per GPU'),
                            ({'gpu_ids': '0,1', 'jobs_per_gpu': '2,0'}, 'greater than zero'),
                            ({'gpu_ids': '0', 'jobs_per_gpu': 'two'}, 'integer'),
                            ({'gpu_ids': '0,0'}, 'duplicate')]:
        with pytest.raises(scheduler.ConfigError, match=message):
            capacity(config)
