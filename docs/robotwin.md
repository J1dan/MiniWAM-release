# RoboTwin 2.0

The RoboTwin model uses action and visual horizons of 24 and executes all 24
actions before replanning. It is evaluated on the 50 RoboTwin 2.0 tasks with
100 episodes each in two conditions: clean scenes with seen instructions, and
randomized scenes with unseen instructions.

## Setup

```bash
bash scripts/setup_robotwin.sh                            # RoboTwin 96c1feab + evaluator patch
bash scripts/setup_robotwin_eval_env.sh --download-assets # simulator env in .robotwin-env/
bash RoboTwin/XPolicyLab/policy/MiniWAM/install.sh miniwam
```

`setup_robotwin.sh` checks out RoboTwin at commit `96c1feab` (XPolicyLab
`c37109c5`), applies `integrations/robotwin/robotwin.patch` and links the
MiniWAM XPolicyLab adapter (`integrations/xpolicylab/policy/MiniWAM`) into it.
The simulator runs in its own conda environment and talks to the MiniWAM policy
server, which runs in the `miniwam` environment. The RoboTwin assets take about
16 GB and the simulator environment about 8 GB; the installer's package cache in
`.cache/robotwin-install/` (about 3 GB) can be deleted afterwards.

## Evaluation protocols

RoboTwin picks evaluation cases online: the evaluator draws seeds and keeps the
ones an expert planner can solve. MiniWAM supports this official protocol
(`configs/robotwin/eval_official.yaml`) and, by default, frozen replay
(`configs/robotwin/eval_frozen.yaml`) of 10,000 cases captured with the official
evaluator. Replay rebuilds each recorded scene, verifies it against the record,
restores the random state and uses the recorded instruction, so it runs the
cases the official evaluator selected and every model sees the same ones.
Compared with the official protocol, frozen replay:

- skips the per-seed expert run, which saves time;
- reads instructions from cached embeddings, so the policy servers do not load
  the UMT5 encoder (about 11 GB of GPU memory each) and more jobs fit on one GPU
  (raise `--jobs-per-gpu`). The cache is checked against the encoder hash recorded
  in the checkpoint, so the encoder file is not needed;
- resumes per episode, while the official protocol restarts an interrupted task.

Download, unpack and verify the frozen cases (9 GB archive):

```bash
hf download J1dan/MiniWAM-RoboTwin-frozen-cases robotwin_eval_bundle_v3_unseen.tar \
  --repo-type dataset --local-dir datasets
# sha256: 618a26cd9ea247d193349afa604207f6956608007f7eba5bb6be7721512aeef5
tar -xf datasets/robotwin_eval_bundle_v3_unseen.tar -C datasets
python scripts/data/package_robotwin_eval_cache.py --verify datasets/robotwin_eval_bundle_v3_unseen
```

The verifier accepts only the files listed in the bundle, so unpack the archive
rather than downloading the folder directly. The archive can be deleted after
verification.

## Evaluate

Run the launcher from the repository root in the `miniwam` environment; it
starts each simulator in `.robotwin-env`. Each job evaluates one task with its
own simulator and policy server, which together use about 7 GB of GPU memory
with frozen replay. `--gpu-ids 0,1` and `--jobs-per-gpu 4` (one value for all
GPUs, or one per GPU, e.g. `2,4`) set where jobs run and how many share a GPU;
they override `gpu_ids` and `jobs_per_gpu` in the eval config.

```bash
for split in demo_clean demo_randomized; do
  python RoboTwin/scripts/eval_policy_multitask.py --config configs/robotwin/eval_frozen.yaml \
    --policy-name MiniWAM --ckpt-name "$PWD/checkpoints/robotwin" \
    --env-cfg-type aloha_agilex --action-type joint \
    --policy-conda-env miniwam --eval-env-conda-env "$PWD/.robotwin-env" \
    --task-config "$split" --test-num 100 --seed 0 \
    --output-dir "$PWD/eval_results/robotwin/$split" --stream-output
done

python scripts/eval/summarize_robotwin.py --evaluation-protocol frozen \
  --clean-summary eval_results/robotwin/demo_clean/<run>/summary.json \
  --randomized-summary eval_results/robotwin/demo_randomized/<run>/summary.json \
  --manifest datasets/robotwin_eval_bundle_v3_unseen/robotwin_eval_manifest.jsonl \
  --text-index datasets/robotwin_eval_bundle_v3_unseen/text_cache/frozen_index.json \
  --output-dir eval_results/robotwin/metrics
```

`--ckpt-name` takes a checkpoint folder, a run folder (its `latest` step is
used) or a path relative to `outputs/`. Rerun the same command to resume:
frozen replay keeps every finished episode in a per-task ledger under
`eval_results/robotwin/frozen_case_ledgers/`. A smaller `--test-num`, such as
20, evaluates the first cases of each task, and a later run with 100 adds the
rest. While a run is in progress, `report.json` in its run folder shows live
progress; `eval_results/robotwin/reports/` keeps the latest report of each
checkpoint.

If rendering fails on a Blackwell GPU, install the newer denoiser with
`bash scripts/setup_robotwin_oidn_blackwell.sh` and add to the eval config:

```yaml
miniwam:
  renderer_compatibility:
    profile: oidn_2_3_3_blackwell
    oidn_root: .robotwin-compat/oidn-2.3.3.x86_64.linux
```

Results from this profile get their own ledger and result namespace.

For the official protocol, run the same loop with
`--config configs/robotwin/eval_official.yaml`, then summarize with
`--evaluation-protocol official` and without `--manifest` and `--text-index`.

### Checking a new machine

The frozen cases depend on RoboTwin's assets and simulator setup. On a new
machine you can confirm, without running the policy, that every recorded scene
is rebuilt exactly (this takes a long time):

```bash
python scripts/eval/audit_robotwin_frozen_manifest.py \
  --manifest datasets/robotwin_eval_bundle_v3_unseen/robotwin_eval_manifest.jsonl \
  --env-prefix "$PWD/.robotwin-env" --task-config both --output-dir eval_results/robotwin/audit
```

`--gpu-ids` and `--jobs-per-gpu` spread its blocks (one per task and condition)
over GPUs, as for evaluation. `audit.json` should report `exact_cases: 10000` and
no mismatches or setup errors. To check a single case, run
`conda run -p .robotwin-env python scripts/eval/check_robotwin_scene_replay.py --task <task> --task-config <condition> --episode-index <i>`;
it needs the simulator environment.

## Training

Extract the 2,500 `demo_clean` training episodes as described in
[data preparation](data.md#robotwin-20), then train both stages with
`configs/robotwin/stage{1,2}.yaml` ([training](training.md)).
`configs/ablations/robotwin/` holds RoboTwin versions of the Wan-VAE and
native-feature controls.
