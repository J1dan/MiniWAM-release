# Contributing

Issues and pull requests are welcome. Before opening a pull request:

- Run the CPU test suite in the `miniwam` environment: `python -m pytest -q`.
  Tests that need a RoboTwin checkout run after `bash scripts/setup_robotwin.sh`.
- Keep new options out of the configs unless a recipe uses them; the configs
  in `configs/` are the paper's recipes.
- A change to model code should keep released checkpoints loading: run one
  LIBERO task with `checkpoints/libero` before and after.

Naming: `MiniWAM` in prose, `miniwam` for the package, the conda environment
and file names. "PRISM" is the learned compact target and its encoder; "world
tokens" are the PRISM tokens denoised by the world expert.

Do not edit `miniwam/robotwin_manifest.py`, `scripts/eval/robotwin_replay_state.py`
or the `scripts/eval_policy_xpolicylab.py` half of
`integrations/robotwin/robotwin.patch`: the frozen RoboTwin cases record their
SHA-256 hashes, and replay refuses to run if they change.
