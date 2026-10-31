"""Resolve explicit MiniWAM RoboTwin evaluation protocol configurations."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

PROTOCOL_ENVIRONMENT = (
    'MINIWAM_EVAL_PROTOCOL', 'MINIWAM_CASE_MODE', 'MINIWAM_EVAL_MANIFEST',
    'MINIWAM_FROZEN_TEXT_INDEX', 'MINIWAM_TEXT_CACHE_MISS',
    'MINIWAM_TEXT_CACHE_DIR', 'MINIWAM_RESULT_NAMESPACE', 'MINIWAM_CASE_OUTPUT',
    'MINIWAM_ROBOTWIN_RENDER_PROFILE', 'MINIWAM_OIDN_ROOT',
)

OIDN_233_PROFILE = 'oidn_2_3_3_blackwell'
# integrations/robotwin/robotwin.patch applied to RoboTwin 96c1feab must give
# exactly the evaluator that captured the certified frozen cases.
OFFICIAL_CAPTURE_SHA256 = 'b76f04ad2a88c99292632c2965bc0687edf3ef2bb66c264c0d8800d673824ec9'


def validate_evaluation_jobs(jobs, config, robotwin_root):
    """Validate both execution modes against the same pinned official contract."""
    import yaml
    from scripts.eval.robotwin_split_contract import BENCHMARK_PROTOCOL, SPLITS
    root = Path(robotwin_root).resolve()
    pins = {root: '96c1feab536306b50c26af200044fcdf126e8904',
            root/'XPolicyLab': 'c37109c500be67d0dea6b36bf7337bbd26e763cd'}
    for repo, expected in pins.items():
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise ValueError(f'Unsupported official source revision: {repo} ({actual})')
    if sha256(root/'scripts/eval_policy_xpolicylab.py') != OFFICIAL_CAPTURE_SHA256:
        raise ValueError('Official evaluator differs from the verified capture/rollout implementation')
    changed = subprocess.check_output(
        ['git', '-C', str(root), 'diff', 'HEAD', '--', 'envs', 'env_cfg', 'description'], text=True)
    if changed:
        raise ValueError('Official task, configuration or instruction sources have local changes')
    changed = subprocess.check_output(
        ['git', '-C', str(root/'XPolicyLab'), 'diff', 'HEAD', '--', 'utils', 'client_server'], text=True)
    if changed:
        raise ValueError('Official XPolicyLab observation/action or transport sources have local changes')
    native = {}
    for setting, expected in SPLITS.items():
        path = root/'env_cfg/task_config'/f'{setting}.yml'
        values = yaml.safe_load(path.read_text())
        if values.get('eval_instruction') != expected or values.get('embodiment') != ['aloha-agilex']:
            raise ValueError(f'Unsupported official task configuration: {setting}')
        native[setting] = dict(instruction_type=expected, config_sha256=sha256(path))
    if config['evaluation_protocol'] == 'frozen':
        from miniwam.robotwin_manifest import load_manifest
        from scripts.eval.robotwin_split_contract import require_marked_contract
        cases, provenance = load_manifest(config['manifest'], root)
        if provenance.get('benchmark_protocol') != BENCHMARK_PROTOCOL:
            raise ValueError('Frozen mode requires the certified official-split release')
        semantic = require_marked_contract(cases, provenance, root)
        if semantic != provenance.get('split_validation'):
            raise ValueError('Frozen release split certification does not match its cases')
    identities = set()
    splits = {}
    from miniwam.robotwin_manifest import content_hash, source_provenance
    from miniwam.robotwin import ROBOTWIN_TASKS
    for job in jobs:
        if job.task_name not in ROBOTWIN_TASKS:
            raise ValueError(f'Task is outside the official 50-task catalog: {job.task_name}')
        overrides = job.overrides
        setting = overrides.get('task_config', 'demo_clean')
        if setting not in native:
            raise ValueError(f'Unsupported benchmark split: {setting}')
        instruction = overrides.get('instruction_type') or native[setting]['instruction_type']
        if instruction != native[setting]['instruction_type']:
            raise ValueError(f'{setting} requires {native[setting]["instruction_type"]} instructions; remove --instruction-type override')
        if overrides.get('eval_batch') or overrides.get('num_workers', 1) != 1:
            raise ValueError('Official case selection requires one simulator worker per task; parallelize task jobs with --jobs-per-gpu')
        if config['evaluation_protocol'] == 'official' and overrides.get('expert_check') is not True:
            raise ValueError('Official evaluation requires --expert-check')
        if job.action_type != 'joint' or job.env_cfg_type != 'aloha_agilex':
            raise ValueError('MiniWAM benchmark evaluation requires aloha_agilex joint actions')
        frequency = overrides.get('frequency')
        if frequency is not None and frequency != 30:
            raise ValueError('MiniWAM official/frozen comparison requires the captured 30 Hz observation contract')
        count = overrides.get('test_num', 100)
        if type(count) is not int or not 1 <= count <= 100:
            raise ValueError('Require 1..100 trials per task; full benchmark evaluation requires 100')
        if type(job.seed) is not int or job.seed < 0:
            raise ValueError('Evaluation seed must be a nonnegative integer')
        if config['evaluation_protocol'] == 'frozen' and job.seed != 0:
            raise ValueError('This frozen release records official seed-0 case selection; use --seed 0')
        identity = (job.task_name, setting, job.seed)
        if identity in identities:
            raise ValueError(f'Duplicate evaluation job: {identity}')
        identities.add(identity)
        splits.setdefault(setting, []).append(dict(task=job.task_name, seed=job.seed, trials=count))
    return dict(benchmark_protocol=BENCHMARK_PROTOCOL, split_configs=native,
                requested_jobs=splits, evaluator_sha256=OFFICIAL_CAPTURE_SHA256,
                case_selection='official_expert_filtering', simulator_workers_per_task=1,
                sources_sha256=content_hash(source_provenance(root)),
                validator_sha256=sha256(__file__))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _path(value, root, name, *, must_exist=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'miniwam.{name} must be a nonempty path string.')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root/path
    path = path.resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(f'miniwam.{name} does not exist: {path}')
    return path


def _resolve_renderer(raw, root):
    """Resolve an explicitly selected renderer compatibility profile.

    Omitting this mapping preserves RoboTwin/SAPIEN's upstream renderer exactly.
    """
    if raw is None:
        return {'profile': 'upstream'}, {
            'MINIWAM_ROBOTWIN_RENDER_PROFILE': 'upstream',
        }, ''
    if not isinstance(raw, dict):
        raise ValueError('miniwam.renderer_compatibility must be a mapping.')
    profile = raw.get('profile')
    allowed = {'profile', 'oidn_root'}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f'Unsupported renderer_compatibility fields: {sorted(unknown)}'
        )
    if profile != OIDN_233_PROFILE:
        raise ValueError(
            f"renderer_compatibility.profile must be '{OIDN_233_PROFILE}'."
        )
    oidn_root = _path(raw.get('oidn_root'), root, 'renderer_compatibility.oidn_root')
    library_dir = oidn_root / 'lib'
    required = (
        library_dir / 'libOpenImageDenoise_core.so.2.3.3',
        library_dir / 'libOpenImageDenoise.so.2.3.3',
        library_dir / 'libOpenImageDenoise_device_cuda.so.2.3.3',
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'OIDN 2.3.3 compatibility installation is incomplete: ' + ', '.join(missing)
        )
    expected = {
        required[0].name: 'eadb2437d6ac4a88e5746980495ba1091f7d14745e3000b82a9382093d22f351',
        required[1].name: '9ac9dcd18318cae879efb01c14cff72c2c4275428a15e5ab6e94c8ff8a43c4dc',
        required[2].name: '12b3ff87cfa4b6a90481b757f5a56ee69380a45e3dcaeadf8dd22cb73d4f714a',
    }
    actual = {path.name: sha256(path) for path in required}
    if actual != expected:
        raise ValueError(f'OIDN 2.3.3 library hashes differ from the pinned release: {actual}')
    normalized = {
        'profile': profile,
        'oidn_root': str(oidn_root),
        'library_sha256': actual,
    }
    environment = {
        'MINIWAM_ROBOTWIN_RENDER_PROFILE': profile,
        'MINIWAM_OIDN_ROOT': str(oidn_root),
    }
    return normalized, environment, f'-renderer-{profile}'


def resolve_protocol_config(raw, root):
    """Return a normalized public config and internal worker environment."""
    root = Path(root).resolve()
    if not isinstance(raw, dict):
        raise ValueError('MiniWAM evaluation requires a top-level miniwam mapping.')
    protocol = raw.get('evaluation_protocol')
    if protocol not in {'frozen', 'official'}:
        raise ValueError("miniwam.evaluation_protocol must be 'frozen' or 'official'.")
    common = {'evaluation_protocol', 'text_cache_dir', 'renderer_compatibility', 'result_namespace'}
    renderer, renderer_environment, renderer_suffix = _resolve_renderer(
        raw.get('renderer_compatibility'), root
    )
    if protocol == 'frozen':
        allowed = common | {'manifest', 'text_index', 'case_output'}
        unknown = set(raw)-allowed
        if unknown:
            raise ValueError(f'Unsupported frozen protocol fields: {sorted(unknown)}')
        manifest = _path(raw.get('manifest'), root, 'manifest', must_exist=True)
        text_index = _path(raw.get('text_index'), root, 'text_index', must_exist=True)
        provenance = manifest.with_suffix('.provenance.json')
        seal = text_index.with_suffix('.sha256.json')
        if not provenance.is_file() or not seal.is_file():
            raise FileNotFoundError('Frozen manifest provenance or text-cache seal is missing.')
        manifest_sha = sha256(manifest)
        recorded = json.loads(provenance.read_text())
        if recorded.get('eval_manifest_sha256') != manifest_sha:
            raise ValueError('Frozen manifest SHA256 differs from its provenance.')
        from scripts.eval.robotwin_split_contract import BENCHMARK_PROTOCOL
        from scripts.data.package_robotwin_eval_cache import validate_v2_certification
        if recorded.get('benchmark_protocol') != BENCHMARK_PROTOCOL or recorded.get('schema_version') != 2:
            raise ValueError('Frozen mode requires the certified official-split release')
        validate_v2_certification(recorded)
        text_index_payload = json.loads(text_index.read_text())
        seal_payload = json.loads(seal.read_text())
        if seal_payload.get('text_cache_sha256') != sha256(text_index):
            raise ValueError('Frozen text-index SHA256 differs from its seal.')
        if text_index_payload.get('eval_manifest_sha256') != manifest_sha:
            raise ValueError('Frozen text index belongs to a different manifest.')
        namespace = raw.get('result_namespace') or f'frozen-{manifest_sha[:12]}{renderer_suffix}'
        cache_dir = _path(raw.get('text_cache_dir', 'datasets/robotwin_eval_text_cache'),
                          root, 'text_cache_dir')
        case_output = raw.get('case_output')
        if case_output is not None:
            case_output = _path(case_output, root, 'case_output')
            if renderer['profile'] != 'upstream':
                case_output = case_output / f"renderer-{renderer['profile']}"
        normalized = dict(evaluation_protocol=protocol, manifest=str(manifest),
                          text_index=str(text_index), text_cache_dir=str(cache_dir),
                          result_namespace=namespace, renderer_compatibility=renderer)
        environment = dict(MINIWAM_EVAL_PROTOCOL=protocol, MINIWAM_CASE_MODE='replay',
            MINIWAM_EVAL_MANIFEST=str(manifest), MINIWAM_FROZEN_TEXT_INDEX=str(text_index),
            MINIWAM_TEXT_CACHE_MISS='error', MINIWAM_TEXT_CACHE_DIR=str(cache_dir),
            MINIWAM_RESULT_NAMESPACE=namespace)
        if case_output is not None:
            normalized['case_output'] = str(case_output)
            environment['MINIWAM_CASE_OUTPUT'] = str(case_output)
        environment.update(renderer_environment)
    else:
        unknown = set(raw)-common
        if unknown:
            raise ValueError(f'Unsupported official protocol fields: {sorted(unknown)}')
        cache_dir = _path(raw.get('text_cache_dir', 'datasets/robotwin_official_text_cache'),
                          root, 'text_cache_dir')
        namespace = f'official{renderer_suffix}'
        normalized = dict(evaluation_protocol=protocol, text_cache_dir=str(cache_dir),
                          result_namespace=namespace, renderer_compatibility=renderer)
        environment = dict(MINIWAM_EVAL_PROTOCOL=protocol, MINIWAM_TEXT_CACHE_MISS='build',
                           MINIWAM_TEXT_CACHE_DIR=str(cache_dir),
                           MINIWAM_RESULT_NAMESPACE=namespace)
        environment.update(renderer_environment)
    return normalized, environment
