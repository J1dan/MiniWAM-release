import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "integrations/xpolicylab/policy/MiniWAM/compat/resume_task.py"
spec = importlib.util.spec_from_file_location("robotwin_resume", MODULE)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def completed(identity="a" * 64, trials=20):
    return (
        f"[MiniWAM] resume_identity={identity}\n"
        f"Final success rate: 0/{trials} = 0%\n"
        "[MiniWAM] task_completed=true\n"
    )


def test_zero_success_is_a_completed_job():
    assert resume.reusable(completed(), "a" * 64, 20)


@pytest.mark.parametrize("text", [
    "step: 192 / 400\n",
    completed().replace("[MiniWAM] task_completed=true\n", ""),
    completed("b" * 64),
    completed(trials=10),
    completed() + "Final success rate: 0/20\n",
    completed().replace("0/20", "21/20"),
])
def test_incomplete_mismatched_or_ambiguous_results_are_not_reused(text):
    assert not resume.reusable(text, "a" * 64, 20)


def test_resume_without_multitask_fails(monkeypatch):
    monkeypatch.delenv("ROBOTWIN_EVAL_ARGS_FILE", raising=False)
    monkeypatch.setenv("MINIWAM_EVAL_RESUME_FROM", "/missing")
    with pytest.raises(ValueError, match="multitask"):
        resume.main([])


def test_resume_replays_complete_job_and_reruns_changed_artifact(tmp_path, monkeypatch, capsys):
    old = tmp_path / "old"
    new = tmp_path / "new"
    (old / "logs").mkdir(parents=True)
    (new / "jobs").mkdir(parents=True)
    args = new / "jobs" / "000_adjust_bottle_seed0.args"
    args.write_text("--test_num\n20\n--task_config\ndemo_clean\n")
    monkeypatch.setenv("ROBOTWIN_EVAL_ARGS_FILE", str(args))
    monkeypatch.delenv("MINIWAM_EVAL_RESUME_FROM", raising=False)
    artifact = ["a" * 64]

    def output(command, **kwargs):
        if command[0] == "bash":
            return (f"[MiniWAM] checkpoint_path={tmp_path}/model.pt\n"
                    f"[MiniWAM] checkpoint_sha256={artifact[0]}\n"
                    f"[MiniWAM] dataset_stats_sha256={'b' * 64}\n")
        if "show" in command:
            return MODULE.read_bytes()
        return "test-revision-or-gpu\n"

    monkeypatch.setattr(resume.subprocess, "check_output", output)
    argv = ["RoboTwin", "adjust_bottle", "checkpoint", "aloha_agilex", "joint",
            "0", "0", "0", "miniwam", "simulator"]
    assert resume.main(argv) == 10
    identity = resume.IDENTITY.findall(capsys.readouterr().out)[0]
    (old / "logs" / "000_adjust_bottle_seed0.log").write_text(
        completed(identity) + f"[MiniWAM] checkpoint_sha256={artifact[0]}\n")
    # Default auto search discovers a matching completed job without an export.
    assert resume.main(argv) == 0
    assert "Reusing completed" in capsys.readouterr().out
    monkeypatch.setenv("MINIWAM_EVAL_RESUME_FROM", str(old))
    assert resume.main(argv) == 0
    assert "Final success rate: 0/20" in capsys.readouterr().out
    artifact[0] = "c" * 64
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        resume.main(argv)
    monkeypatch.delenv("MINIWAM_EVAL_RESUME_FROM")
    assert resume.main(argv) == 10
    artifact[0] = "a" * 64
    monkeypatch.setenv("MINIWAM_EVAL_RESUME_FROM", "off")
    assert resume.main(argv) == 10
