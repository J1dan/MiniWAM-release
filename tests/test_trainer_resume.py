"""Data-order position of a resumed training run."""
from scripts.train import resume_position


def test_resume_position_continues_epoch_and_skips_consumed_batches():
    assert resume_position(0, 1, 100) == (0, 0)
    assert resume_position(250, 1, 100) == (2, 50)
    assert resume_position(60, 4, 100) == (2, 40)
    assert resume_position(100, 1, 100) == (1, 0)
