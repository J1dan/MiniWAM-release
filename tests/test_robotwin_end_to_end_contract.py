import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from miniwam import robotwin_features as features
from miniwam.models.dinov3 import DINOv3FrameEncoder
from scripts.data.extraction_utils import encode_dinov3_frames
from scripts.eval.summarize_robotwin import validate_setting_evidence
from scripts.data import extract_dataset_feature_robotwin as extraction


def test_uint8_dark_pixels_have_same_scale_alone_or_batched():
    encoder = DINOv3FrameEncoder(backbone=torch.nn.Linear(1, 1))
    dark = torch.ones(1, 3, 16, 16, dtype=torch.uint8)
    bright = torch.full_like(dark, 255)
    alone = encoder.preprocess(dark)
    together = encoder.preprocess(torch.cat([dark, bright]))[:1]
    assert torch.equal(alone, together)
    expected = (torch.full((1, 3, 224, 224), 1 / 255) - encoder.fallback_pixel_mean) / encoder.fallback_pixel_std
    torch.testing.assert_close(alone, expected)
    class Boundary:
        def encode_for_miniwam(self, value):
            assert value.dtype == torch.uint8
            return value.to(torch.bfloat16)
    assert torch.equal(encode_dinov3_frames(dark, Boundary(), 'cpu', 1), dark.to(torch.bfloat16))


def test_training_root_rejects_missing_duplicate_and_mixed_cases(tmp_path, monkeypatch):
    # Reduce only the case catalog; numeric statistics validation has its own tests.
    monkeypatch.setattr(features, 'ROBOTWIN_TASKS', ('task',))
    monkeypatch.setattr(features, 'ROBOTWIN_EPISODE_IDS', ('0000000', '0000001'))
    monkeypatch.setattr(features, 'validate_robotwin_training_statistics', lambda _: None)
    stats = dict(stats_id='stats', source_manifest_sha256='source', aggregation=dict(transitions=2))
    monkeypatch.setattr(features, 'load_statistics_mapping', lambda _: stats)
    monkeypatch.setattr(features, 'select_statistics', lambda *a: stats)
    (tmp_path/'extraction_audit.json').write_text(json.dumps(dict(selected_episode_count=2500,
        source_manifest_sha256='source', extraction=dict(complete=True, stats_id='stats'))))
    contract = dict(dinov3={'processor': {'size': 224}}, text={'encoding': 'frozen'})
    metas = []
    for index in range(2):
        directory = tmp_path/f'episode_{index:06d}'; directory.mkdir()
        meta = dict(task_name='task', source_episode_id=f'{index:07d}', stats_id='stats',
                    dataset_name=features.ROBOTWIN_DATASET_KEY, main_camera_keys=['cam_head'],
                    wrist_camera_keys=['cam_left_wrist', 'cam_right_wrist'], num_raw_frames=1,
                    dinov3=contract['dinov3'], text_contract=contract['text'])
        path = directory/'metadata.json'; path.write_text(json.dumps(meta)); metas.append((path, meta))
        for name in ('dino_features.pt', 'text_embed.pt', 'actions.npy', 'proprio.npy'):
            (directory/name).touch()
    assert features.validate_training_feature_root(tmp_path) == contract
    path, meta = metas[1]
    path.write_text(json.dumps(dict(meta, source_episode_id='0000000')))
    with pytest.raises(ValueError, match='duplicate'):
        features.validate_training_feature_root(tmp_path)
    path.write_text(json.dumps(dict(meta, dinov3={'processor': {'size': 256}})))
    with pytest.raises(ValueError, match='Mixed'):
        features.validate_training_feature_root(tmp_path)
    unbound = dict(meta); del unbound['text_contract']; path.write_text(json.dumps(unbound))
    with pytest.raises(ValueError, match='verify-existing-text'):
        features.validate_training_feature_root(tmp_path)
    path.write_text(json.dumps(meta)); (path.parent/'proprio.npy').unlink()
    with pytest.raises(FileNotFoundError):
        features.validate_training_feature_root(tmp_path)


def test_training_root_accepts_one_strict_wan_vae_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(features, 'ROBOTWIN_TASKS', ('task',))
    monkeypatch.setattr(features, 'ROBOTWIN_EPISODE_IDS', ('0000000',))
    monkeypatch.setattr(features, 'validate_robotwin_training_statistics', lambda _: None)
    stats = dict(stats_id='stats', source_manifest_sha256='source', aggregation=dict(transitions=1))
    monkeypatch.setattr(features, 'load_statistics_mapping', lambda _: stats)
    monkeypatch.setattr(features, 'select_statistics', lambda *a: stats)
    (tmp_path/'extraction_audit.json').write_text(json.dumps(dict(
        selected_episode_count=2500, source_manifest_sha256='source',
        extraction=dict(complete=True, stats_id='stats'),
    )))
    directory = tmp_path/'episode_000000'; directory.mkdir()
    vae_contract = {'checkpoint_sha256': 'hash', 'future_rgb_horizon': 16}
    (directory/'metadata.json').write_text(json.dumps(dict(
        task_name='task', source_episode_id='0000000', stats_id='stats',
        dataset_name=features.ROBOTWIN_DATASET_KEY,
        main_camera_keys=['cam_head'],
        wrist_camera_keys=['cam_left_wrist', 'cam_right_wrist'],
        num_raw_frames=1, visual_encoder_type='wan_vae', wan_vae=vae_contract,
        text_contract={'encoding': 'frozen'},
    )))
    for name in ('vae_latent.pt', 'text_embed.pt', 'actions.npy', 'proprio.npy'):
        (directory/name).touch()
    assert features.validate_training_feature_root(
        tmp_path, expected_visual_type='wan_vae',
    ) == {'wan_vae': vae_contract, 'text': {'encoding': 'frozen'}}
    with pytest.raises(ValueError, match='config requests'):
        features.validate_training_feature_root(tmp_path, expected_visual_type='dinov3')


def test_metrics_require_actual_case_and_provenance_evidence(tmp_path):
    (tmp_path/'jobs').mkdir(); artifacts = tmp_path/'artifacts'; artifacts.mkdir()
    (tmp_path/'jobs/task.artifacts.json').write_text(json.dumps(dict(path=str(artifacts))))
    expected = dict(eval_manifest_sha256='manifest', text_cache_sha256='cache',
                    robotwin_commit='r', xpolicylab_commit='x')
    full = dict(expected, checkpoint_sha256='checkpoint', dataset_stats_sha256='stats')
    case = dict(task_name='task', task_config='demo_clean', accepted_seed=100001,
                episode_index=0, instruction='exact instruction')
    record = dict(case, success=False, provenance=full)
    report = dict(full, results=[record], total_episodes=1, total_successes=0)
    path = artifacts/'summary.json'; path.write_text(json.dumps(report))
    rows = [dict(task='task', setting='demo_clean', checkpoint_sha256='checkpoint',
                 statistics_sha256='stats', successes=0, log=str(tmp_path/'task.log'))]
    validate_setting_evidence(tmp_path/'summary.json', rows, [case], expected, 1)
    record['instruction'] = 'different instruction'; path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='case/provenance'):
        validate_setting_evidence(tmp_path/'summary.json', rows, [case], expected, 1)
    record['instruction'] = case['instruction']; report['text_cache_sha256'] = 'other'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='provenance'):
        validate_setting_evidence(tmp_path/'summary.json', rows, [case], expected, 1)


def test_trainer_binds_the_feature_contract(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'configs/robotwin/stage1.yaml').read_text())
    cfg['pretrained'] = yaml.safe_load((root/'configs/base.yaml').read_text())['pretrained']
    contract = dict(dinov3={}, text=dict(encoding=dict(tokenizer_id='google/umt5-xxl', context_len=128)))
    monkeypatch.setattr(
        features, 'validate_training_feature_root',
        lambda _, expected_visual_type=None: contract,
    )
    features.bind_training_features(cfg, tmp_path)
    assert cfg['data']['feature_contract'] == contract
    cfg['data']['context_len'] = 64
    with pytest.raises(ValueError, match='context_len'):
        features.bind_training_features(cfg, tmp_path)
    # A generic/LIBERO config must not acquire the RoboTwin-specific constraint.
    features.bind_training_features({'data': {'dataset_name': 'libero'}}, tmp_path)
    assert 'bind_training_features(cfg, features_root)' in (root/'scripts/train.py').read_text()


def test_existing_text_binding_requires_exact_source_and_features(tmp_path, monkeypatch):
    from types import SimpleNamespace
    directory = tmp_path/'episode_000000'; directory.mkdir()
    meta = dict(task_name='task', source_episode_id='0000000', stats_id='stats',
        image_decoder='XPolicyLab.utils.process_data.decode_image_bit', image_color_order='RGB',
        action_dim=14, proprio_dim=14, action_type='joint', action_representation='absolute_qpos',
        action_transform=None, action_normalization=extraction.ACTION_ZSCORE,
        proprio_normalization=extraction.PROPRIO_ZSCORE, dinov3={'model': 'dino'},
        language_instruction='Exact instruction.', num_raw_frames=1)
    metadata = directory/'metadata.json'; metadata.write_text(json.dumps(meta))
    for name in ('actions.npy', 'proprio.npy'):
        np.save(directory/name, np.zeros((1, 14), dtype=np.float32))
    context = torch.zeros(128, 2, dtype=torch.bfloat16)
    mask = torch.ones(128, dtype=torch.bool)
    torch.save(dict(context=context, mask=mask), directory/'text_embed.pt')
    monkeypatch.setattr(extraction, 'validate_multiview_episode', lambda *a, **kw: None)
    monkeypatch.setattr(extraction, 'encode_text', lambda *a: (context.clone(), mask.clone()))
    source = {'instruction': 'Exact instruction.'}
    options = dict(output_dir=tmp_path, load_xspark_v1=lambda *a, **kw: source,
        decode_image_bit=None, pack_robot_state=None,
        visual_encoder_type='dinov3', vae=None,
        dino_encoder=SimpleNamespace(feature_metadata=lambda **kw: {'model': 'dino'}),
        visual_contract={'model': 'dino'}, future_rgb_horizon=16,
        per_view_resolution=(224, 224),
        text_encoder=None, tokenizer=None, device='cpu', text_dtype=torch.bfloat16,
        batch_size=1, stats={'stats_id': 'stats'}, text_contract={'encoder': 'frozen'})
    episode = extraction.RoboTwinEpisode('task', '0000000', tmp_path/'source.hdf5')
    with pytest.raises(ValueError, match='lacks text provenance'):
        extraction.process_episode(0, episode, **options)
    options['verify_existing_text'] = True
    source['instruction'] = 'Different instruction.'
    with pytest.raises(ValueError, match='instruction differs'):
        extraction.process_episode(0, episode, **options)
    source['instruction'] = 'Exact instruction.'
    torch.save(dict(context=context + 1, mask=mask), directory/'text_embed.pt')
    with pytest.raises(ValueError, match='embedding differs'):
        extraction.process_episode(0, episode, **options)
    assert 'text_contract' not in json.loads(metadata.read_text())
    torch.save(dict(context=context, mask=mask), directory/'text_embed.pt')
    assert extraction.process_episode(0, episode, **options) is False
    assert json.loads(metadata.read_text())['text_contract'] == options['text_contract']


def test_corrupt_statistics_are_not_silently_recomputed(tmp_path):
    path = tmp_path/'dataset_statistics.pkl'
    assert extraction._load_matching_statistics(path, {}) is None
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Invalid existing statistics'):
        extraction._load_matching_statistics(path, {})
    assert path.read_bytes() == b'corrupted'


def test_driver_library_override_precedes_probe_and_vulkan_is_scoped(tmp_path):
    root = Path(__file__).resolve().parents[1]
    helper = root/'integrations/xpolicylab/policy/MiniWAM/compat/configure_nvidia_runtime.sh'
    probe = tmp_path/'nvidia-smi'
    probe.write_text('#!/bin/bash\n[[ "${LD_LIBRARY_PATH}" == "${MINIWAM_NVIDIA_LIBRARY_PATH}"* ]]\n')
    probe.chmod(0o755)
    icd = tmp_path/'icd.json'; icd.write_text('{}')
    env = dict(os.environ, PATH=f'{tmp_path}:/usr/bin:/bin',
               MINIWAM_NVIDIA_LIBRARY_PATH=str(tmp_path), MINIWAM_VULKAN_ICD_FILENAMES=str(icd))
    env.pop('VK_ICD_FILENAMES', None)
    command = 'source "$1" "$2" || exit; printf "%s" "${VK_ICD_FILENAMES:-unset}"'
    def run(role, values):
        return subprocess.run(['bash', '-c', command, 'test', str(helper), role],
                              env=values, capture_output=True, text=True)
    result = run('simulator', env)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(icd)
    result = run('policy', env)
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'unset'
    for name in ('MINIWAM_NVIDIA_LIBRARY_PATH', 'MINIWAM_VULKAN_ICD_FILENAMES'):
        result = run('simulator', dict(env, **{name: env[name] + ':'}))
        assert result.returncode != 0 and 'empty entry' in result.stderr
