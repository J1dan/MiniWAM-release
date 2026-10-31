#!/usr/bin/env python3
"""Validate frozen pre-policy scene reconstruction without a policy rollout."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', default='beat_block_hammer')
    p.add_argument('--task-config', default='demo_randomized')
    p.add_argument('--episode-index', type=int, default=0)
    p.add_argument('--manifest', type=Path,
                   default=ROOT/'datasets/robotwin_eval_bundle_v3_unseen/robotwin_eval_manifest.jsonl')
    p.add_argument('--expert-reference', action='store_true',
                   help='Also compare direct setup with setup after one official expert trajectory.')
    p.add_argument('--audit-release-assets', action='store_true',
                   help='Check all release-v1 sampled model configs against local assets without simulation.')
    args = p.parse_args()
    args.manifest = args.manifest.resolve()
    root = ROOT/'RoboTwin'
    os.chdir(root)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('official_eval', root/'scripts/eval_policy_xpolicylab.py')
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    spec = importlib.util.spec_from_file_location('protocol', ROOT/'miniwam/robotwin_manifest.py')
    protocol = importlib.util.module_from_spec(spec); spec.loader.exec_module(protocol)
    spec = importlib.util.spec_from_file_location(
        'frozen_hooks', ROOT/'integrations/xpolicylab/policy/MiniWAM/compat/frozen_cases.py')
    frozen_hooks = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen_hooks)
    cfg, _ = official.load_task_args(dict(task_name=args.task, task_config=args.task_config, policy_name='SCENE_AUDIT'))
    cfg['eval_mode'] = True
    cases, _ = protocol.load_manifest(args.manifest, root)
    if args.audit_release_assets:
        audited = 0
        choices = 0
        for frozen_case in cases:
            overrides = frozen_hooks.release_model_choice_overrides(frozen_case, root)
            if overrides:
                audited += 1
                choices += len(overrides)
        print(f'PASS: {audited} affected cases resolve {choices} model choices uniquely '
              'against local assets', flush=True)
        return
    matches = [case for case in cases if case['task_name'] == args.task and
               case['task_config'] == args.task_config and
               case['episode_index'] == args.episode_index]
    if len(matches) != 1:
        raise ValueError(f'Expected one frozen case, found {len(matches)}')
    case = matches[0]
    env = official.class_decorator(args.task)
    hashes = []
    dynamic_hashes = []
    validation_modes = []
    def setup_frozen_case():
        replay = frozen_hooks.ReleaseModelChoiceReplay(case, root)
        replay.install()
        complete = False
        try:
            env.setup_demo(now_ep_num=case['episode_index'], seed=case['accepted_seed'],
                           is_test=True, **cfg)
            complete = True
        finally:
            replay.restore(require_complete=complete)
    def dynamics():
        # Additional paired checks beyond the recorded manifest snapshot.
        from sapien.physx import PhysxRigidDynamicComponent
        return protocol.content_hash(dict(
            actors=sorted([dict(name=a.get_name(), velocity=body.linear_velocity,
                                angular_velocity=body.angular_velocity)
                           for a in env.scene.get_all_actors()
                           for body in a.components if isinstance(body, PhysxRigidDynamicComponent)],
                          key=lambda a: a['name']),
            articulations=sorted([dict(name=a.get_name(), qvel=a.get_qvel())
                                  for a in env.scene.get_all_articulations()], key=lambda a: a['name']),
            eval_success=env.eval_success, take_action_cnt=env.take_action_cnt,
            step_lim=env.step_lim, plan_success=env.plan_success))
    for repeat in range(2):
        snapshot = None
        try:
            setup_frozen_case()
            snapshot = protocol.scene_snapshot(
                env, case['scene_snapshot']['episode_info']
            )
            validation_mode, digest = frozen_hooks.validate_release_scene(case, snapshot)
            hashes.append(digest)
            validation_modes.append(validation_mode)
            dynamic_hashes.append(dynamics())
            print(f'Reconstruction {repeat}: {digest} ({validation_mode})', flush=True)
        except ValueError:
            if snapshot is not None:
                protocol.atomic_json(ROOT/'datasets/scene_replay_mismatch.json', dict(
                    case=case, reconstructed_scene=snapshot,
                    reconstructed_sha256=protocol.content_hash(snapshot)))
            raise
        finally:
            official.safe_close_env(env, clear_cache=True)
    if len(set(hashes)) != 1:
        raise ValueError('Repeated official setup produced different scene fingerprints')
    if len(set(dynamic_hashes)) != 1:
        raise ValueError('Repeated setup produced different velocities or rollout counters')
    if len(set(validation_modes)) != 1:
        raise ValueError('Repeated setup used different scene-validation modes')
    if args.expert_reference:
        try:
            setup_frozen_case()
            episode_info = env.play_once()
            env.close_env()
            expert_passed = bool(env.plan_success and env.check_success())
            print(f'Expert reference passed filtering: {expert_passed}', flush=True)
            setup_frozen_case()
            reference = protocol.scene_snapshot(env, case['scene_snapshot']['episode_info'])
            frozen_hooks.validate_release_scene(case, reference)
            if dynamics() != dynamic_hashes[0]:
                raise ValueError('Post-expert reset differs from direct frozen setup')
            if expert_passed and protocol.canonical(episode_info) != case['scene_snapshot']['episode_info']:
                raise ValueError('Expert instruction metadata differs from frozen case')
            print('PASS: post-expert and direct setup match, including velocities and rollout counters', flush=True)
        finally:
            official.safe_close_env(env, clear_cache=True)
    print('PASS: exact frozen scene reconstruction without expert re-execution', flush=True)


if __name__ == '__main__':
    main()
