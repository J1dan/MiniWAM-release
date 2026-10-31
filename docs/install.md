# Installation

MiniWAM runs on Linux x86-64 with Python 3.10, PyTorch 2.11 and an NVIDIA GPU
from the Turing generation or newer. The PyTorch wheels bundle the CUDA 12.8
runtime, so no CUDA toolkit is needed and any CUDA 12 driver works: 525 or
newer, or 570 or newer for Blackwell GPUs. We tested with driver 580. Run all
commands from the repository root.

## Environment

```bash
sudo apt-get install -y git git-lfs ffmpeg libegl1 libgl1 libglib2.0-0 libosmesa6
bash scripts/setup_miniwam_env.sh
conda activate miniwam
python scripts/verify_miniwam_environment.py --require-cuda --check-libero
```

`setup_miniwam_env.sh` creates the `miniwam` conda environment from
`environment.yml`, installs this package with `pip install -e .`, clones LIBERO
at a pinned revision into `LIBERO/` and writes `.libero/config.yaml`, so that
LIBERO does not prompt for its paths. Pass `--skip-libero` on a machine that
only trains or evaluates RoboTwin. A new Miniconda installation must first
accept the terms of service of Anaconda's default channels; until then, conda
stops and prints the commands that accept them.

## Model assets

DINOv3 is gated: accept its license on Hugging Face and run `hf auth login`.

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --revision 5931719e67bbdb9737e363e781fb0c67687896bc
hf download google/umt5-xxl --include '*token*' '*.model'
```

The UMT5-XXL text encoder (11 GB) is needed to extract training features, to
build instruction caches and for the official RoboTwin protocol. Evaluating the
released checkpoints does not need it: LIBERO and LIBERO-Plus read the released
instruction caches, and frozen RoboTwin replay checks its cache against the
encoder hash recorded in the checkpoint. (`--check-assets` below still expects
the file.)

```bash
hf download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth \
  --revision b62153b54fce61f5908de23f6d8715e147de0daf --local-dir pretrained/language
# sha256: 7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d
```

The Wan2.1 VAE is only needed for the Wan-VAE ablations:

```bash
hf download Wan-AI/Wan2.1-VACE-14B Wan2.1_VAE.pth \
  --revision ab1867ace0fb7738d46c46aab8308b71c51bb011 --local-dir pretrained/vision
# sha256: 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981
```

Unmodified copies of the UMT5 encoder and the VAE, with their licenses, are also
in the [checkpoint repository](https://huggingface.co/J1dan/MiniWAM);
`hf download J1dan/MiniWAM --include 'pretrained/*' --local-dir .` puts both
where MiniWAM expects them.

Check the assets with
`python scripts/verify_miniwam_environment.py --require-cuda --check-assets --visual-encoder dinov3`
(or `wan_vae`).

## Benchmarks

- **LIBERO** is installed by `setup_miniwam_env.sh`.
- **LIBERO-Plus** needs its own checkout and assets; see
  [evaluation](evaluation.md#libero-plus).
- **RoboTwin 2.0** runs the simulator in a separate conda environment; see
  [RoboTwin](robotwin.md).

Keep the pinned revisions installed by the setup scripts: the evaluation
protocols and the frozen RoboTwin cases depend on them.
