"""Semantic release gate, separate from historical scene replay certification."""
import hashlib
from pathlib import Path

import yaml

BENCHMARK_PROTOCOL = 'official_split_v1'
SPLITS = {'demo_clean': 'seen', 'demo_randomized': 'unseen'}


def validate_split_contract(cases, provenance, robotwin_root):
    """Require official split labels and the exact pinned task configurations."""
    root = Path(robotwin_root)
    hashes = provenance['sources']['source_hashes']
    configs = {}
    for split, expected in SPLITS.items():
        name = f'env_cfg/task_config/{split}.yml'
        payload = (root/name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if hashes.get(name) != digest:
            raise ValueError(f'Official split config differs from capture: {name}')
        if yaml.safe_load(payload).get('eval_instruction') != expected:
            raise ValueError(f'Official instruction setting changed: {split}')
        configs[split] = dict(instruction_type=expected, config_sha256=digest)
    counts = {split: 0 for split in SPLITS}
    for case in cases:
        split = case['task_config']
        if split not in SPLITS or case['instruction_type'] != SPLITS[split]:
            raise ValueError(f'Official split/instruction mismatch: {split}/{case["instruction_type"]}')
        counts[split] += 1
    return dict(benchmark_protocol=BENCHMARK_PROTOCOL, split_configs=configs,
                cases_per_split=counts, valid_cases=sum(counts.values()))


def require_marked_contract(cases, provenance, robotwin_root):
    marker = provenance.get('benchmark_protocol')
    if marker is None:
        return None  # Historical releases certified replay only.
    if marker != BENCHMARK_PROTOCOL:
        raise ValueError(f'Unknown benchmark protocol: {marker}')
    return validate_split_contract(cases, provenance, robotwin_root)
