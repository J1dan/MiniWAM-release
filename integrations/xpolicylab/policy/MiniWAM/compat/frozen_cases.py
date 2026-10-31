"""Frozen-case replay hooks called by the patched official single-worker evaluator.

The cases were captured with the official evaluator (including its expert check).
Replay rebuilds each captured scene, verifies it, restores the captured RNG state
and records the policy's success in a per-task ledger.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import importlib.util
import tempfile

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.eval.robotwin_replay_state import clutter_defaults, normalize_snapshot
spec = importlib.util.spec_from_file_location('miniwam_manifest_protocol', ROOT/'miniwam/robotwin_manifest.py')
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
atomic_json, case_key, load_manifest, sha256 = (
    protocol.atomic_json, protocol.case_key, protocol.load_manifest, protocol.sha256)

# Hooks that captured released manifests; the second is the one recorded by the
# certified 493d9b60 release.
RELEASED_CAPTURE_HOOK_SHA256 = {
    'b24d7d437f5f6d07372ea4f67ee9454793c80a265dacb5e0ab3566cc8b38b8c3',
    '242f27c27834c10370ffb26bd67b83eda91348dedb8b7a5299784124121dbea8',
}

RELEASE_ASSET_SELECTIONS = {
    'place_a2b_left': (
        ('selected_modelname_A', 'object'),
        ('selected_modelname_B', 'target_object'),
    ),
    'place_a2b_right': (
        ('selected_modelname_A', 'object'),
        ('selected_modelname_B', 'target_object'),
    ),
    'place_object_scale': (('selected_modelname', 'object'),),
    'place_object_stand': (('selected_modelname', 'object'),),
    'put_object_cabinet': (('selected_modelname', 'object'),),
}
STALE_PLANNER_ALIAS_TASKS = {'stack_blocks_two', 'stack_blocks_three'}


def release_model_choice_overrides(case, root):
    """Recover captured model IDs from exact model-data stored in release v1."""
    overrides = []
    snapshot = case['scene_snapshot']
    for identity_key, model_key in RELEASE_ASSET_SELECTIONS.get(case['task_name'], ()):
        model_name = snapshot['identities'].get(identity_key)
        expected_config = snapshot['models'].get(model_key)
        if not isinstance(model_name, str) or not isinstance(expected_config, dict):
            raise ValueError(
                f'Frozen case lacks {identity_key}/{model_key} needed for portable asset replay'
            )
        candidates = {}
        for path in sorted((Path(root)/'assets'/'objects'/model_name).glob('model_data*.json')):
            match = re.fullmatch(r'model_data(\d+)\.json', path.name)
            if match:
                candidates[int(match.group(1))] = path
        matches = [model_id for model_id, path in candidates.items()
                   if protocol.canonical(json.loads(path.read_text())) == expected_config]
        if len(matches) != 1:
            raise ValueError(
                f'Expected exactly one local model-data match for {model_name}/{model_key}; '
                f'found IDs {matches}'
            )
        overrides.append((frozenset(candidates), matches[0], model_name, model_key))
    return overrides


class ReleaseModelChoiceReplay:
    """Force release-v1 model choices without changing the NumPy RNG stream."""
    def __init__(self, case, root):
        self.pending = release_model_choice_overrides(case, root)
        self.original = None
        self.setup_defaults = clutter_defaults(case)
        self.saved_defaults = None

    def install(self):
        if self.setup_defaults is not None:
            if self.saved_defaults is not None:
                raise RuntimeError('Frozen setup state is already installed')
            from envs._base_task import Base_Task
            self.saved_defaults = Base_Task.get_cluttered_table.__defaults__
            Base_Task.get_cluttered_table.__defaults__ = self.setup_defaults
        if not self.pending:
            return
        import numpy as np
        if self.original is not None:
            raise RuntimeError('Frozen model-choice replay is already installed')
        self.original = np.random.choice

        def choice(values, *args, **kwargs):
            # Always execute the official draw first, preserving its exact RNG advance.
            result = self.original(values, *args, **kwargs)
            array = np.asarray(values)
            if self.pending and array.ndim == 1:
                candidate_ids, expected_id, model_name, model_key = self.pending[0]
                try:
                    observed_ids = frozenset(int(value) for value in array.tolist())
                except (TypeError, ValueError):
                    observed_ids = frozenset()
                if observed_ids == candidate_ids:
                    if kwargs.get('size') is not None or (args and args[0] is not None):
                        raise ValueError('Frozen model-ID choice unexpectedly requested a sample array')
                    self.pending.pop(0)
                    return array.dtype.type(expected_id)
            return result

        np.random.choice = choice

    def restore(self, require_complete=True):
        if self.saved_defaults is not None:
            from envs._base_task import Base_Task
            Base_Task.get_cluttered_table.__defaults__ = self.saved_defaults
            self.saved_defaults = None
        if self.original is None:
            if require_complete and self.pending:
                raise ValueError('Frozen setup did not consume expected model-ID choices')
            return
        import numpy as np
        np.random.choice = self.original
        self.original = None
        if require_complete and self.pending:
            missing = [(name, key, expected) for _, expected, name, key in self.pending]
            raise ValueError(f'Frozen setup did not consume model-ID choices: {missing}')


def validate_release_scene(case, reconstructed):
    """Return the exact/narrow compatibility mode or fail on a scene mismatch."""
    if case.get('scene_normalization'):
        reconstructed = normalize_snapshot(case, reconstructed)
    reconstructed_hash = protocol.content_hash(reconstructed)
    if reconstructed_hash == case['scene_sha256']:
        return 'exact_pre_policy_scene_sha256', reconstructed_hash
    if case['task_name'] in STALE_PLANNER_ALIAS_TASKS:
        expected_semantic = json.loads(json.dumps(case['scene_snapshot']))
        reconstructed_semantic = json.loads(json.dumps(reconstructed))
        expected_semantic.get('models', {}).pop('last_actor', None)
        reconstructed_semantic.get('models', {}).pop('last_actor', None)
        if protocol.content_hash(expected_semantic) == protocol.content_hash(reconstructed_semantic):
            return 'release_v1_ignore_stale_planner_last_actor', reconstructed_hash
    raise ValueError('Reconstructed scene differs from frozen case')


def write_replay_ledger(path, records):
    """Commit a whole task ledger atomically while the task lock is held."""
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class FrozenCases:
    def __init__(self, args, usr_args, save_dir, instruction_type, test_num):
        self.mode = os.environ.get('MINIWAM_CASE_MODE', '')
        self.enabled = bool(self.mode)
        self.capture = False  # capturing new cases is not part of the public release
        self.replay = self.mode == 'replay'
        self.count = test_num
        self.current = None
        self.current_validated = False
        self.model_choice_replay = None
        self.current_scene_validation = None
        if not self.enabled:
            return
        if self.mode != 'replay':
            raise ValueError('MINIWAM_CASE_MODE must be replay')
        if usr_args.get('eval_batch') or int(usr_args.get('num_workers', 1)) != 1:
            raise ValueError('Frozen cases require one official simulator worker per task job')
        self.task = args['task_name']
        self.setting = args['task_config']
        self.instruction_type = instruction_type
        self.root = Path(usr_args['root_dir']).resolve()
        self.save_dir = Path(save_dir)
        effective = {k: v for k, v in args.items() if k not in
                     {'policy_name', 'ckpt_setting', 'eval_video_save_dir'}}
        effective['action_type'] = usr_args.get('action_type', 'joint')
        effective['frequency'] = usr_args.get('frequency', usr_args.get('ctrl_freq', 30))
        self.setup_hash = protocol.content_hash(protocol.portable_config(effective, self.root))
        self.output = Path(os.environ['MINIWAM_CASE_OUTPUT'])
        self.output.mkdir(parents=True, exist_ok=True)
        # Prevent two jobs racing on the same task/setting ledger.
        self.lock = (self.output/f'{self.task}__{self.setting}.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases, provenance = load_manifest(os.environ['MINIWAM_EVAL_MANIFEST'], self.root)
        if provenance.get('replay_state_sha256') is not None:
            if provenance['replay_state_sha256'] != sha256(ROOT/'scripts/eval/robotwin_replay_state.py'):
                raise ValueError('Replay setup-state protocol hash mismatch')
            if provenance.get('release_status') != 'certified_exact_local_replay':
                raise ValueError('Replay-preserving release is not certified')
        capture_hook = provenance['generation_script_sha256']
        if capture_hook not in RELEASED_CAPTURE_HOOK_SHA256:
            raise ValueError('Unknown frozen-case capture hook hash')
        if provenance['scene_protocol_sha256'] != sha256(ROOT/'miniwam/robotwin_manifest.py'):
            raise ValueError('Scene fingerprint protocol hash mismatch')
        cache_index = Path(os.environ['MINIWAM_FROZEN_TEXT_INDEX'])
        cache = json.loads(cache_index.read_text())
        seal = json.loads(cache_index.with_suffix('.sha256.json').read_text())
        if sha256(cache_index) != seal['text_cache_sha256']:
            raise ValueError('Text cache index hash mismatch')
        if cache['eval_manifest_sha256'] != provenance['eval_manifest_sha256']:
            raise ValueError('Text cache belongs to a different manifest')
        for entry in cache['entries']:
            if sha256(cache_index.parent/entry['file']) != entry['embedding_sha256']:
                raise ValueError('Text cache entry hash mismatch')
        checkpoint = Path(os.environ['MINIWAM_CHECKPOINT_PATH'])
        self.provenance = dict(checkpoint_sha256=sha256(checkpoint/'model.safetensors'),
            dataset_stats_sha256=sha256(checkpoint/'statistics.json'),
            eval_manifest_sha256=provenance['eval_manifest_sha256'],
            text_cache_sha256=sha256(cache_index),
            robotwin_commit=provenance['sources']['robotwin_commit'],
            xpolicylab_commit=provenance['sources']['xpolicylab_commit'])
        self.path = self.output/f'{self.task}__{self.setting}.jsonl'
        self.completed = [json.loads(s) for s in self.path.read_text().splitlines()] if self.path.exists() else []
        if any(c['provenance'] != self.provenance for c in self.completed):
            raise ValueError('Result ledger provenance mismatch')
        done = {case_key(c) for c in self.completed}
        if len(done) != len(self.completed):
            raise ValueError('Duplicate completed cases')
        canonical = {case_key(c): c for c in cases}
        if any(case_key(c) not in canonical or any(c[k] != v for k, v in canonical[case_key(c)].items())
               or type(c['success']) is not bool for c in self.completed):
            raise ValueError('Result ledger case differs from manifest')
        selected = [c for c in cases if c['task_name'] == self.task and c['task_config'] == self.setting
                    and c['episode_index'] < test_num]
        if len(selected) != test_num or any(c['instruction_type'] != instruction_type for c in selected):
            raise ValueError('Manifest selection/count/instruction type mismatch')
        # Check even when every selected case is already completed and
        # accepted() will not execute during this invocation.
        if any(c['setup_sha256'] != self.setup_hash for c in selected):
            raise ValueError('Effective simulator setup differs from capture')
        if any(c['task_name'] != self.task or c['task_config'] != self.setting
               for c in self.completed):
            raise ValueError('Result ledger belongs to a different task/setting')
        self.selected = {case_key(c) for c in selected}
        self.pending = [c for c in selected if case_key(c) not in done]
        self.count = len(self.pending)
        self.replay_metadata = dict(
            **self.provenance,
            renderer_compatibility=self.renderer_metadata(),
            case_selection='official_expert_filtering_at_manifest_capture',
            replay_expert_planner_executed=False,
            replay_scene_validation='exact_sha256_or_release_v1_stale_planner_alias',
            release_asset_replay='manifest_model_config_to_model_id_preserve_numpy_rng',
            replay_hook_sha256=sha256(__file__),
        )
        atomic_json(self.save_dir/'metadata.json', self.replay_metadata)
        args_file = os.environ.get('ROBOTWIN_EVAL_ARGS_FILE')
        if args_file:
            # Partial attempts may never print the official final-result
            # log line. Bind their artifact directory before rollout.
            atomic_json(Path(args_file).with_suffix('.artifacts.json'),
                        dict(path=str(self.save_dir.resolve())))
        # eval_remote_policy reads this after FrozenCases is constructed.
        # Accepted seeds are already fixed by the manifest; rerunning the
        # expert trajectory can reject them nondeterministically.
        usr_args['expert_check'] = False

    @staticmethod
    def renderer_metadata():
        profile = os.environ.get('MINIWAM_ROBOTWIN_RENDER_PROFILE', 'upstream')
        metadata = {'profile': profile}
        if profile == 'oidn_2_3_3_blackwell':
            root = Path(os.environ['MINIWAM_OIDN_ROOT']).resolve()
            names = (
                'libOpenImageDenoise_core.so.2.3.3',
                'libOpenImageDenoise.so.2.3.3',
                'libOpenImageDenoise_device_cuda.so.2.3.3',
            )
            metadata.update(
                oidn_version='2.3.3',
                library_sha256={name: sha256(root/'lib'/name) for name in names},
            )
        return metadata

    def seed(self, index, proposed):
        if self.replay:
            self.current = self.pending[index]
            self.current_validated = False
            self.current_scene_validation = None
            self.model_choice_replay = ReleaseModelChoiceReplay(self.current, self.root)
            self.model_choice_replay.install()
            return self.current['accepted_seed']
        return proposed

    def instruction(self):
        return self.current['instruction']

    def accepted(self, seed, index, instruction, episode_info, task_env):
        if self.model_choice_replay is not None:
            self.model_choice_replay.restore(require_complete=True)
            self.model_choice_replay = None
        self.current_validated = False
        if seed != self.current['accepted_seed'] or instruction != self.current['instruction']:
            raise ValueError('Frozen case identity/instruction mismatch')
        snapshot = protocol.scene_snapshot(task_env, self.current['scene_snapshot']['episode_info'])
        snapshot_hash = protocol.content_hash(snapshot)
        if not isinstance(instruction, str) or not instruction.strip() or instruction == self.task:
            raise ValueError('Official instruction generation failed or fell back to task name')
        if self.current.get('setup_sha256') != self.setup_hash:
            raise ValueError('Effective simulator setup differs from capture')
        try:
            validation, _ = validate_release_scene(self.current, snapshot)
        except ValueError:
            atomic_json(self.save_dir/'scene_mismatch.json', dict(case=self.current,
                reconstructed_scene=snapshot, reconstructed_sha256=snapshot_hash))
            raise ValueError(
                f'Reconstructed scene differs from frozen case: {self.task}/{self.setting}/{seed}'
            )
        import numpy as np
        import random
        state = self.current['numpy_rng_state']
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        def tuples(value):
            return tuple(map(tuples, value)) if isinstance(value, list) else value
        random.setstate(tuples(self.current['python_rng_state']))
        self.current_validated = True
        self.current_scene_validation = validation

    def result(self, success):
        if self.replay:
            if not self.current_validated or type(success) is not bool:
                raise ValueError('Cannot record an unvalidated case or non-boolean success')
            if any(case_key(c) == case_key(self.current) for c in self.completed):
                raise ValueError('Duplicate completed case')
            record = dict(self.current, success=success, provenance=self.provenance,
                          replay_hook_sha256=self.replay_metadata['replay_hook_sha256'],
                          scene_validation=self.current_scene_validation,
                          replay_expert_planner_executed=False)
            write_replay_ledger(self.path, [*self.completed, record])
            self.completed.append(record)
            self.current_validated = False
            self.current_scene_validation = None
            atomic_json(self.save_dir/'summary.json', dict(**self.replay_metadata,
                total_episodes=len(self.completed), total_successes=sum(c['success'] for c in self.completed),
                results=self.completed))

    def finish(self):
        results = [c for c in self.completed if case_key(c) in self.selected]
        if len(results) != len(self.selected):
            raise ValueError('Frozen evaluation ended with missing cases')
        atomic_json(self.save_dir/'summary.json', dict(**self.replay_metadata,
            total_episodes=len(results), total_successes=sum(c['success'] for c in results), results=results))
        return sum(c['success'] for c in results)
