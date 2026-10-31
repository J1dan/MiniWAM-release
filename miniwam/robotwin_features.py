"""Training feature provenance and the matching deployment contract."""
import hashlib
import json
from pathlib import Path

from miniwam.robotwin import (
    ROBOTWIN_TASKS, ROBOTWIN_EPISODE_IDS, ROBOTWIN_DATASET_KEY,
    ROBOTWIN_ACTION_DIM_INFO, validate_robotwin_training_statistics,
    assert_robotwin_data_config, assert_robotwin_joint_contract,
)
from miniwam.statistics import load_statistics_mapping, select_statistics


def wan_vae_feature_contract(checkpoint_path, resolution, dtype, future_rgb_horizon):
    """Return the sealed preprocessing/feature contract for RoboTwin Wan-VAE."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Wan-VAE checkpoint does not exist: {checkpoint}')
    digest = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    size = [int(value) for value in resolution]
    if len(size) != 2 or any(value <= 0 for value in size):
        raise ValueError(f'Wan-VAE resolution must be [H,W], got {resolution!r}.')
    horizon = int(future_rgb_horizon)
    if horizon <= 0 or horizon % 4:
        raise ValueError('Wan-VAE future_rgb_horizon must be positive and divisible by 4.')
    dtype_name = str(dtype).removeprefix('torch.')
    return {
        'architecture': 'WanVideoVAE',
        'checkpoint_filename': checkpoint.name,
        'checkpoint_sha256': digest.hexdigest(),
        'z_dim': 16,
        'compute_dtype': dtype_name,
        'storage_dtype': 'bfloat16',
        'input_color_order': 'RGB',
        'input_dtype': 'uint8',
        'input_range': [-1.0, 1.0],
        'resize': {
            'size_hw': size,
            'mode': 'bilinear',
            'align_corners': False,
            'crop': None,
        },
        'temporal_encoding': 'joint_current_future_v1',
        'future_rgb_horizon': horizon,
        'temporal_downsample_factor': 4,
    }


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def text_feature_contract(encoder_path, tokenizer_id, tokenizer, context_len, dtype, cache_dir):
    from miniwam.robotwin_text_cache import RoboTwinTextCache
    cache = RoboTwinTextCache(cache_dir, encoder_path, tokenizer_id, context_len, dtype,
                             miss_policy='build', build_device='cuda')
    backend = json.loads(tokenizer.tokenizer.backend_tokenizer.to_str())
    return dict(encoding=cache.contract, tokenizer_backend_sha256=json_sha256(backend))


def validate_runtime_text_contract(expected, cache, *, require_frozen=True):
    from miniwam.robotwin_text_cache import comparable_contract

    if comparable_contract(expected['encoding']) != cache.contract:
        raise ValueError('Evaluation text encoder/prompt/dtype differs from training extraction.')
    if require_frozen:
        if cache.frozen_entries is None:
            raise ValueError('Frozen RoboTwin evaluation requires a frozen text index.')
        backend = json.loads((cache.directory/'tokenizer/tokenizer.json').read_text())
        if json_sha256(backend) != expected['tokenizer_backend_sha256']:
            raw_backend = dict(backend)
            raw_backend['padding'] = None
            raw_backend['truncation'] = None
            if json_sha256(raw_backend) != expected['tokenizer_backend_sha256']:
                raise ValueError('Frozen evaluation tokenizer differs from training extraction.')
    elif cache.expected_tokenizer_backend_sha256 != expected['tokenizer_backend_sha256']:
        raise ValueError('Official evaluation tokenizer binding differs from training extraction.')


def validate_training_feature_root(root, expected_visual_type=None):
    """Require complete, unique 50x50 features and one extraction contract."""
    root = Path(root)
    audit = json.loads((root/'extraction_audit.json').read_text())
    if (audit.get('selected_episode_count') != 2500 or
            audit.get('extraction', {}).get('complete') is not True):
        raise ValueError('RoboTwin training requires a completed 2500-episode extraction audit.')
    stats_path = root/'dataset_statistics.pkl'
    stats = select_statistics(load_statistics_mapping(stats_path), ROBOTWIN_DATASET_KEY, stats_path)
    validate_robotwin_training_statistics(stats)
    if (stats['source_manifest_sha256'] != audit['source_manifest_sha256'] or
            audit['extraction']['stats_id'] != stats['stats_id']):
        raise ValueError('RoboTwin extraction audit/statistics provenance mismatch.')
    expected = {(task, episode) for task in ROBOTWIN_TASKS for episode in ROBOTWIN_EPISODE_IDS}
    seen, contract, transitions, visual_type = set(), None, 0, None
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and
                            (p.name.startswith('episode_') or (p/'metadata.json').exists())):
        meta = json.loads((directory/'metadata.json').read_text())
        key = (meta.get('task_name'), meta.get('source_episode_id'))
        if key not in expected or key in seen:
            raise ValueError(f'Unexpected or duplicate RoboTwin training case: {directory}: {key}')
        seen.add(key)
        if (meta.get('dataset_name') != ROBOTWIN_DATASET_KEY or
                meta.get('stats_id') != stats['stats_id'] or
                meta.get('main_camera_keys') != ['cam_head'] or
                meta.get('wrist_camera_keys') != ['cam_left_wrist', 'cam_right_wrist']):
            raise ValueError(f'RoboTwin feature schema/statistics mismatch: {directory}')
        if 'text_contract' not in meta:
            raise ValueError(f'{directory} has text features without a text contract. Run extraction with '
                             '--verify-existing-text to compare and bind every existing embedding.')
        episode_visual_type = meta.get('visual_encoder_type')
        if episode_visual_type is None:
            episode_visual_type = 'dinov3' if 'dinov3' in meta else 'wan_vae'
        if episode_visual_type not in {'dinov3', 'wan_vae'}:
            raise ValueError(f'Unsupported RoboTwin visual encoder in {directory}: {episode_visual_type!r}')
        if visual_type is not None and episode_visual_type != visual_type:
            raise ValueError(f'Mixed RoboTwin visual feature types: {directory}')
        visual_type = episode_visual_type
        visual_key = 'dinov3' if visual_type == 'dinov3' else 'wan_vae'
        if visual_key not in meta:
            raise ValueError(f'{directory} is missing its {visual_key} feature contract.')
        current = {visual_key: meta[visual_key], 'text': meta['text_contract']}
        if contract is not None and current != contract:
            raise ValueError(f'Mixed visual/text extraction contracts: {directory}')
        contract = current
        visual_file = 'dino_features.pt' if visual_type == 'dinov3' else 'vae_latent.pt'
        for name in (visual_file, 'text_embed.pt', 'actions.npy', 'proprio.npy'):
            if not (directory/name).is_file():
                raise FileNotFoundError(directory/name)
        transitions += int(meta['num_raw_frames'])
    if seen != expected or transitions != stats['aggregation']['transitions']:
        raise ValueError('RoboTwin feature cases/transition count differ from the exact training split.')
    if expected_visual_type is not None and visual_type != expected_visual_type:
        raise ValueError(
            f'RoboTwin config requests visual_encoder.type={expected_visual_type!r}, '
            f'but feature root contains {visual_type!r}.'
        )
    return contract


def bind_training_features(config, features_root):
    """Apply the same RoboTwin checks in DDP and FSDP before model construction."""
    data = config['data']
    if data.get('dataset_name') != ROBOTWIN_DATASET_KEY:
        return
    assert_robotwin_data_config(data)
    assert_robotwin_joint_contract(action_type=data.get('action_type'),
        robot_action_dim_info=ROBOTWIN_ACTION_DIM_INFO, action_dim=data.get('action_dim'),
        proprio_dim=data.get('proprio_dim'), action_representation=data.get('action_representation'),
        action_transform=data.get('action_transform'), delta_action_mask=data.get('delta_action_mask'))
    visual_type = config.get('model', {}).get('visual_encoder', {}).get('type', 'wan_vae')
    contract = validate_training_feature_root(features_root, expected_visual_type=visual_type)
    pretrained = config.get('pretrained', {})
    if visual_type == 'wan_vae':
        vae_contract = contract['wan_vae']
        if int(vae_contract['future_rgb_horizon']) != int(data['future_rgb_horizon']):
            raise ValueError(
                'Training future_rgb_horizon differs from the horizon-bound Wan-VAE cache.'
            )
        visual = config.get('model', {}).get('visual_encoder', {})
        vae_path = visual.get('vae_path') or data.get('vae_path') or pretrained.get('vae_path')
        if not vae_path:
            raise ValueError(
                'RoboTwin Wan-VAE training requires visual_encoder.vae_path, '
                'data.vae_path, or pretrained.vae_path.'
            )
        actual_vae = wan_vae_feature_contract(
            vae_path,
            vae_contract['resize']['size_hw'],
            vae_contract['compute_dtype'],
            data['future_rgb_horizon'],
        )
        if actual_vae != vae_contract:
            raise ValueError('Training Wan-VAE asset/preprocessing differs from extracted features.')
        expected_grid = [int(value) // 8 for value in vae_contract['resize']['size_hw']]
        if list(visual.get('feature_grid', expected_grid)) != expected_grid:
            raise ValueError(
                f'Training Wan-VAE feature_grid must be {expected_grid} for the extracted '
                f'resolution, got {visual.get("feature_grid")!r}.'
            )
    encoding = contract['text']['encoding']
    for name, default in (('tokenizer_id', None), ('context_len', 128)):
        requested = data.get(name) or pretrained.get(name, default)
        if requested != encoding[name]:
            raise ValueError(f'Training {name} differs from extracted text contract.')
    data['feature_contract'] = contract
