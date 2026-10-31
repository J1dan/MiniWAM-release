# LIBERO and LIBERO-Plus evaluation

Evaluation loads a checkpoint folder: the released `checkpoints/libero`, or a
training output such as `outputs/libero_stage2/latest`. Normalization
statistics come from the folder; nothing is recomputed. Results go to
`eval_results/`, one folder per benchmark (`libero/`, `libero_plus/`,
`robotwin/`). RoboTwin 2.0 is covered in [its own guide](robotwin.md).

## LIBERO

```bash
# one task, two episodes
python scripts/eval/eval_libero_miniwam.py --checkpoint checkpoints/libero \
  --task_suite_name libero_10 --task_id 0 --num_trials 2 --output_dir eval_results/libero_smoke

# all four suites: 40 tasks × 50 episodes
python scripts/eval/run_libero_parallel_miniwam.py --checkpoint checkpoints/libero \
  --task_suite_names libero_10,libero_spatial,libero_object,libero_goal \
  --num_trials 50 --gpu_ids 0,1,2,3 --output_dir eval_results/libero
```

The parallel launcher runs one task per GPU at a time (`--max_tasks_per_gpu`
raises this, with one value for all GPUs or one per GPU, e.g. `8,4`) and retries
a task with less concurrency if it runs out of memory. Each task uses about
2.2 GB of GPU memory.
Rerunning the same command resumes; finished episodes are kept. Each task
writes `summary.json` and rollout videos, and the launcher writes an overall
report.

Instructions are read from `libero_instruction_features.pt` in the checkpoint
folder. For a checkpoint you trained, pass
`--text_feature_cache datasets/dino_features_libero_aggregate_minmax/libero_instruction_features.pt`
or copy that file into the folder; `--online_text_encoding` encodes instructions
with UMT5 instead. The released cache was written during feature extraction;
online encoding can differ from it slightly.

Protocol (`common_libero_v1`, shared with the LIBERO-Plus evaluator and the
baselines in the paper): environment seed 0; episode *i* starts from LIBERO's
initial state *i*; Python, NumPy, PyTorch and policy seeds are reset to 42 at
every episode; the robot settles for 10 no-op steps, then the policy has 220,
280, 300 or 520 steps for spatial, object, goal or LIBERO-10. It predicts
16-step action chunks with 20 denoising steps, executes 10 actions and replans.
Inference runs in BF16. Each `summary.json` records these values under
`evaluation_protocol`, and resuming refuses results recorded under another
protocol.

## LIBERO-Plus

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) perturbs LIBERO tasks
(camera, robot state, language, lighting, background, noise, layout) into 10,030
tasks, each run once with the LIBERO checkpoint.

### Setup

```bash
sudo apt-get install -y libexpat1 libfontconfig1-dev libmagickwand-dev unzip
bash scripts/setup_libero_plus.sh            # pinned fork in LIBERO-plus/
hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset \
  --revision dd2bd61b7d9a6fef1abc52d606e983b41886a149 --local-dir datasets/libero_plus_assets
# sha256: 96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf
unzip -q datasets/libero_plus_assets/assets.zip -d datasets/libero_plus_assets/extracted
mv datasets/libero_plus_assets/extracted/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets \
  LIBERO-plus/libero/libero/assets
python scripts/verify_miniwam_environment.py --require-cuda --check-libero-plus
```

Do not install the fork's `requirements.txt`; its three extra packages are
already pinned in `environment.yml`. LIBERO and LIBERO-Plus both provide the
`libero` package, so keep the two checkouts separate; the evaluator checks that
every task is loaded from `LIBERO-plus/`.

### Instruction cache

LIBERO-Plus appends perturbation metadata to some task names (for example
`... view 0 0 100 0 0 initstate 384`). MiniWAM encodes the clean instruction: the
original LIBERO wording, or the paraphrase for language perturbations. The
released caches are in the
[checkpoint repository](https://huggingface.co/J1dan/MiniWAM):

```bash
hf download J1dan/MiniWAM --include 'libero_plus_text_features/*' --local-dir datasets
# sha256: 9c5467e1cfa69a7e2dcb747ae569549679d640b75c76108cc8a094e8c0941a59  libero_plus_full_10030.pt
#         c67048f85bf030a42060bd03c017fe09ec9e82ff61b1c0eb0e65ccd72be5e7d9  libero_plus_2k.pt
```

To build them instead (needs UMT5 and a GPU):

```bash
python scripts/data/build_libero_plus_text_feature_cache.py \
  --libero_plus_root LIBERO-plus --libero_root LIBERO \
  --baseline_text_feature_cache checkpoints/libero/libero_instruction_features.pt \
  --output datasets/libero_plus_text_features/libero_plus_full_10030.pt \
  --device cuda --dtype bf16
```

Add `--task_manifest scripts/eval/manifests/libero_plus_ablation_2k_seed42.txt`
for the 2K subset. The builder encodes every instruction with UMT5 and checks
the unperturbed ones against the LIBERO cache in the checkpoint, allowing the
small numerical differences between GPU types; it fails on a wrong prompt
template, tokenizer or encoder. On an RTX 3090 the builder reproduces the
embeddings of both released caches bit for bit.

### Run

```bash
python scripts/eval/run_libero_plus_parallel_miniwam.py --checkpoint checkpoints/libero \
  --text_feature_cache datasets/libero_plus_text_features/libero_plus_full_10030.pt \
  --gpu_ids 0,1,2,3 --output_dir eval_results/libero_plus
```

Each GPU runs `--max_tasks_per_gpu` persistent workers (one value for all GPUs,
or one per GPU, e.g. `8,4`); each worker loads the model once, uses about 2.2 GB
of GPU memory and evaluates its share of the tasks one at a time. `--resume` continues an interrupted run in
the same output directory and retries failed tasks. `report.md` breaks the
success rate down by suite, perturbation category and difficulty; execution
errors are reported separately and never counted as policy failures.
`--save_videos sample10` records a fixed ~10% of tasks.

Protocol: environment seed 0, initial state 0, policy seed 42, 10 settling steps,
and 220 / 280 / 300 / 520 policy steps for Spatial / Object / Goal / LIBERO-10.

### LIBERO-Plus-2K

The ablations in the paper (Tables 1 and 4) use a fixed 2,000-task subset,
`scripts/eval/manifests/libero_plus_ablation_2k_seed42.txt`, stratified by
suite, perturbation category and difficulty. Pass it with `--task_manifest` and
use the 2K instruction cache. `python scripts/data/generate_libero_plus_ablation_subset.py --check`
confirms that the manifest regenerates byte-for-byte from the pinned fork.
