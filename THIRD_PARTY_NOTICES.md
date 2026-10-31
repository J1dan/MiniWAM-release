# Third-party notices

MiniWAM includes code adapted from the projects below. Each keeps its license;
source files name their origin. The Apache License 2.0 text is in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

| Code in this repository | Origin | License |
| --- | --- | --- |
| `miniwam/models/dit_block.py`, `mot.py`, `action_dit.py`, `scheduler.py`; structure of `video_dit.py` and `world_dit.py` | [FastWAM](https://github.com/yuantianyuan01/FastWAM), Copyright (c) 2026 The FastWAM Authors | MIT |
| DiT building blocks in `miniwam/models/dit_block.py` (via FastWAM) | [Wan2.1](https://github.com/Wan-Video/Wan2.1) DiT as ported in [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), Copyright 2024-2025 The Alibaba Wan Team Authors | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `miniwam/models/wan_video_vae.py`, `wan_video_text_encoder.py` | [Wan2.1](https://github.com/Wan-Video/Wan2.1) VAE and UMT5 encoder, Copyright 2024-2025 The Alibaba Wan Team Authors, via DiffSynth-Studio and FastWAM; the VAE is trimmed to the Wan2.1 classes | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Action-chunk padding in `miniwam/dataset.py`; `scripts/eval/libero_utils.py` | [OpenVLA](https://github.com/openvla/openvla) / Prismatic, Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti | MIT |
| `quat2axisangle` in `scripts/eval/libero_utils.py` | [robosuite](https://github.com/ARISE-Initiative/robosuite), Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab | MIT |

## Downloaded separately

These are not part of this repository. Setup scripts fetch them at pinned
revisions, and they keep their own terms.

- Benchmarks: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (MIT),
  [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus),
  [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) and XPolicyLab.
  `integrations/robotwin/robotwin.patch` modifies three RoboTwin evaluator files.
- Models: [DINOv3](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
  (DINOv3 License), [UMT5-XXL](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) and the
  [Wan2.1 VAE](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) (Apache 2.0).
- Data: [LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
  (CC BY 4.0, derived from LIBERO) and the RoboTwin 2.0 demonstrations.

## Redistributed copies

The MiniWAM authors did not create the models below. Unmodified copies, each
with its upstream license, are kept under `pretrained/` in the
[checkpoint repository](https://huggingface.co/J1dan/MiniWAM) so that the exact
revisions behind the results stay available:

- The UMT5-XXL encoder (Google, bf16 export by the Wan Team) from
  `Wan-AI/Wan2.1-T2V-14B` at `b62153b5`, under Apache 2.0.
- The Wan2.1 VAE (Wan Team, Alibaba) from `Wan-AI/Wan2.1-VACE-14B` at
  `ab1867ac`, under Apache 2.0.

The instruction embeddings in the same repository are UMT5-XXL outputs for
LIBERO (MIT) and LIBERO-Plus instructions. The frozen RoboTwin evaluation cases in
[J1dan/MiniWAM-RoboTwin-frozen-cases](https://huggingface.co/datasets/J1dan/MiniWAM-RoboTwin-frozen-cases)
were recorded with the RoboTwin 2.0 evaluator (MIT). They include UMT5-XXL
instruction embeddings and the `google/umt5-xxl` tokenizer (Apache 2.0).
