"""Frozen RoboTwin case protocol. Standard library only (also used by simulator)."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def canonical(value):
    """Stable JSON values; reject unsupported values instead of hashing repr/address."""
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError('Fingerprint mapping keys must be strings')
        return {k: canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if hasattr(value, 'tolist'):
        return canonical(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported fingerprint value: {type(value).__name__}')


def content_hash(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def portable_config(value, root):
    if isinstance(value, dict):
        return {k: portable_config(v, root) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable_config(v, root) for v in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and value.startswith(str(Path(root).resolve()) + '/'):
        return '$ROBOTWIN_ROOT/' + value[len(str(Path(root).resolve())) + 1:]
    return canonical(value)


def scene_snapshot(env, episode_info):
    """Post-setup state, before instruction application; no images or RNG draws."""
    def pose(item):
        p = item.get_pose()
        return dict(p=p.p, q=p.q)
    actors = [dict(name=a.get_name(), pose=pose(a)) for a in env.scene.get_all_actors()]
    articulations = [dict(name=a.get_name(), pose=pose(a), qpos=a.get_qpos(),
                         links=[dict(name=l.get_name(), pose=pose(l)) for l in a.get_links()])
                     for a in env.scene.get_all_articulations()]
    cameras = [dict(name=c.get_name(), pose=pose(c), intrinsic=c.get_intrinsic_matrix(),
                    width=c.get_width(), height=c.get_height()) for c in env.scene.get_cameras()]
    lights = [dict(kind=kind, pose=pose(light), color=light.get_color())
              for kind, lights in [('directional', env.direction_light_lst), ('point', env.point_light_lst)]
              for light in lights]
    # Actor wrappers retain model metadata, including scale/contact geometry.
    models = {}
    def model_metadata(name, value):
        if type(value).__module__ == 'envs.utils.actor_utils' and hasattr(value, 'config'):
            models[name] = canonical(value.config)
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                model_metadata(f'{name}[{i}]', child)
    for name, value in vars(env).items():
        model_metadata(name, value)
    identities = {name: canonical(value) for name, value in vars(env).items()
                  if (name.endswith(('_id', '_ids')) or 'modelname' in name or name in ('color_name', 'color_value'))
                  and isinstance(value, (str, int, float, bool, list, tuple))}
    result = canonical(dict(schema_version=1, episode_info=episode_info,
        actors=actors, articulations=articulations, cameras=cameras, lights=lights,
        models=models, identities=identities, setup_info=env.info, table_z_bias=env.table_z_bias,
        wall_texture=env.wall_texture, table_texture=env.table_texture))
    # Engine allocation IDs and insertion order are not case identity.
    for key in ('actors', 'articulations', 'cameras', 'lights'):
        result[key].sort(key=lambda v: json.dumps(v, sort_keys=True))
    return result


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def source_provenance(root):
    root = Path(root).resolve()
    def commit(path):
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    # Hash working-tree sources too: commit IDs alone do not describe local patches.
    files = sorted(set(root.glob('env_cfg/task_config/*.yml')) |
                   set(root.glob('envs/**/*.py')) | set(root.glob('description/**/*.json')) |
                   set(root.glob('description/**/*.py')) |
                   {root/'scripts/eval_policy_xpolicylab.py'})
    files = sorted(set(files) | set(root.glob('XPolicyLab/utils/**/*.py')) |
                   set(root.glob('XPolicyLab/client_server/**/*.py')) |
                   set(root.glob('assets/**/config.yml')))
    return dict(robotwin_commit=commit(root), xpolicylab_commit=commit(root/'XPolicyLab'),
                source_hashes={str(p.relative_to(root)): sha256(p) for p in files})


def case_key(case):
    return (case['task_name'], case['task_config'], case['accepted_seed'])


def validate_cases(cases, tasks, per_setting=100):
    expected = {(task, setting) for task in tasks for setting in ('demo_clean', 'demo_randomized')}
    counts = Counter()
    keys = set()
    indices = {}
    for c in cases:
        key = case_key(c)
        if key in keys:
            raise ValueError(f'Duplicate case: {key}')
        keys.add(key)
        group = key[:2]
        if (group not in expected or type(c['accepted_seed']) is not int or
                type(c['episode_index']) is not int or
                c['instruction_type'] not in ('seen', 'unseen') or
                not isinstance(c['instruction'], str) or not c['instruction'].strip()):
            raise ValueError(f'Invalid manifest case: {c}')
        counts[group] += 1
        indices.setdefault(group, set()).add(c['episode_index'])
    if set(counts) != expected or any(counts[g] != per_setting or
            indices[g] != set(range(per_setting)) for g in expected):
        raise ValueError(f'Expected exactly {per_setting} indexed cases per task/setting; got {dict(counts)}')


def load_manifest(path, root=None):
    path = Path(path)
    provenance = json.loads(path.with_suffix('.provenance.json').read_text())
    if sha256(path) != provenance['eval_manifest_sha256']:
        raise ValueError('Evaluation manifest SHA256 mismatch')
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    validate_cases(cases, provenance['tasks'])
    for case in cases:
        if content_hash(case['scene_snapshot']) != case['scene_sha256']:
            raise ValueError('Manifest scene snapshot hash mismatch')
    if len(cases) != 10000 or len(provenance['tasks']) != 50:
        raise ValueError('Production manifest must contain 50 tasks and 10000 cases')
    if root is not None and source_provenance(root) != provenance['sources']:
        raise ValueError('RoboTwin/XPolicyLab source provenance mismatch')
    return cases, provenance
