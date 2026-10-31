import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'integrations/xpolicylab/policy/MiniWAM/compat/prepare_job.py'
spec = importlib.util.spec_from_file_location('prepare_job', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('settings', [
    {'--num_workers': '2'}, {'--eval_batch': 'true'},
])
def test_shared_workers_rejected(settings):
    with pytest.raises(ValueError, match='isolated parallel'):
        module.validate_workers(settings)


def test_schedule_pins_once_even_if_latest_moves(tmp_path, monkeypatch, capsys):
    jobs = tmp_path / 'jobs'
    jobs.mkdir()
    args_file = jobs / 'task.args'
    args_file.write_text('--num_workers\n1\n--eval_batch\nfalse\n')
    monkeypatch.setenv('ROBOTWIN_EVAL_ARGS_FILE', str(args_file))
    monkeypatch.delenv('MINIWAM_CHECKPOINT_PATH', raising=False)
    from scripts.eval.robotwin_protocol import sha256
    (tmp_path / 'schedule_plan.json').write_text(json.dumps({'miniwam': {
        'evaluation_protocol': 'frozen',
        'benchmark_validation': {
            'benchmark_protocol': 'official_split_v1',
            'validator_sha256': sha256(Path('scripts/eval/robotwin_protocol.py')),
        },
    }}))
    monkeypatch.setenv('MINIWAM_EVAL_PROTOCOL', 'frozen')
    calls = []

    def resolve(*args, **kwargs):
        calls.append(1)
        return '[MiniWAM] checkpoint_path=/fixed/step_034304/model.pt\n[MiniWAM] checkpoint_sha256=abc\n'

    monkeypatch.setattr(module.subprocess, 'check_output', resolve)
    argv = ['RoboTwin', 'task', 'run/latest', 'aloha_agilex', 'joint', '0', '0', '0', 'miniwam', 'sim']
    module.main(argv)
    first = capsys.readouterr().out
    module.main([argv[0], 'another_task', *argv[2:]])
    assert capsys.readouterr().out == first
    assert len(calls) == 1
    assert (tmp_path / 'miniwam_checkpoint.json').exists()
    argv[2] = 'different_checkpoint'
    with pytest.raises(ValueError, match='request changed'):
        module.main(argv)
