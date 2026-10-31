import json
import pytest

from scripts.eval.robotwin_runtime_identity import bind_frozen_runtime


def test_resume_rejects_changed_policy_execution(tmp_path):
    identity = dict(checkpoint_sha256='same', replan_steps=24, eval_manifest_sha256='new')
    bind_frozen_runtime(tmp_path, identity)
    bind_frozen_runtime(tmp_path, dict(identity))
    with pytest.raises(ValueError, match='identity changed'):
        bind_frozen_runtime(tmp_path, dict(identity, replan_steps=16))
    assert json.loads((tmp_path/'runtime_identity.json').read_text()) == identity


def test_results_without_runtime_identity_are_not_resumed(tmp_path):
    (tmp_path/'adjust_bottle__demo_clean.jsonl').write_text('{"success": true}\n')
    with pytest.raises(ValueError, match='no recorded runtime identity'):
        bind_frozen_runtime(tmp_path, dict(replan_steps=24))
