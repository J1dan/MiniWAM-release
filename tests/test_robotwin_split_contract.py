import hashlib

import pytest

from scripts.eval.robotwin_split_contract import (
    BENCHMARK_PROTOCOL, SPLITS, require_marked_contract, validate_split_contract,
)


@pytest.fixture
def official(tmp_path):
    hashes = {}
    for split, kind in SPLITS.items():
        name = f'env_cfg/task_config/{split}.yml'
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f'eval_instruction: {kind}\n'.encode()
        path.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return tmp_path, dict(sources=dict(source_hashes=hashes))


def test_correct_split_and_pinned_configs(official):
    root, provenance = official
    cases = [dict(task_config=s, instruction_type=t) for s, t in SPLITS.items()]
    assert validate_split_contract(cases, provenance, root)['valid_cases'] == 2


def test_rejects_randomized_with_seen_instructions(official):
    root, provenance = official
    with pytest.raises(ValueError, match='split/instruction mismatch'):
        validate_split_contract([dict(task_config='demo_randomized', instruction_type='seen')], provenance, root)


def test_rejects_changed_native_config(official):
    root, provenance = official
    (root/'env_cfg/task_config/demo_randomized.yml').write_text('eval_instruction: seen\n')
    with pytest.raises(ValueError, match='differs from capture'):
        validate_split_contract([], provenance, root)


def test_rejects_pinned_but_wrong_native_config(official):
    root, provenance = official
    name = 'env_cfg/task_config/demo_randomized.yml'
    payload = b'eval_instruction: seen\n'
    (root/name).write_bytes(payload)
    provenance['sources']['source_hashes'][name] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match='setting changed'):
        validate_split_contract([], provenance, root)


def test_split_is_checked_only_for_marked_releases(official):
    root, provenance = official
    wrong = [dict(task_config='demo_randomized', instruction_type='seen')]
    assert require_marked_contract(wrong, provenance, root) is None
    provenance['benchmark_protocol'] = BENCHMARK_PROTOCOL
    with pytest.raises(ValueError, match='split/instruction mismatch'):
        require_marked_contract(wrong, provenance, root)


def test_unknown_marker_rejected(official):
    root, provenance = official
    provenance['benchmark_protocol'] = 'unknown'
    with pytest.raises(ValueError, match='Unknown benchmark protocol'):
        require_marked_contract([], provenance, root)
