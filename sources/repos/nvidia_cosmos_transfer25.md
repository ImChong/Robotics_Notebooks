# nvidia-cosmos/cosmos-transfer2.5

> 来源归档

- **标题：** Cosmos-Transfer2.5
- **类型：** repo
- **组织：** nvidia-cosmos
- **代码：** <https://github.com/nvidia-cosmos/cosmos-transfer2.5>
- **论文：** <https://arxiv.org/abs/2511.00062>
- **文档：** <https://docs.nvidia.com/cosmos/latest/transfer2.5/index.html>
- **权重：** <https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B>
- **Stars：** ~733（2026-09-05）
- **入库日期：** 2026-09-05
- **一句话说明：** 建在 Predict2.5 上的 **多 ControlNet 世界翻译** 仓：JSON 配 depth / edge / seg / blur；支持当场算控制图、多卡、蒸馏 Edge、驾驶多视角与机器人多视角。
- **沉淀到 wiki：** 是 → [`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（代码 Apache-2.0；权重 NVIDIA Open Model License） |
| **代码** | <https://github.com/nvidia-cosmos/cosmos-transfer2.5> |
| **权重** | <https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B>（含 `distilled/general/edge`、`auto`） |
| **Cookbook** | <https://nvidia-cosmos.github.io/cosmos-cookbook/> |
| **维护** | README **Important**（2026-06）：不再积极开发；新 transfer 能力在 [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos)（Cosmos 3） |

## README / docs 要点（2026-09-05）

- 两条增广：仿真→照片级；真机 RGB→世界状态多样性（天气、外观）。
- 模型族：`Cosmos-Transfer2.5-2B`（通用）、`/auto`（驾驶多视角）、`/robot-multiview-control`（四控）。
- 推理入口：`examples/inference.py -i <spec.json>`；多卡 `torchrun --nproc_per_node=8`。
- 单卡 2B 约 **65.4 GB** VRAM；720p/16fps/93 帧 segmentation：B200 扩散 ~92 s，E2E（121 帧两 chunk）~187 s。
- 蒸馏 Edge：4 step；单卡相对 base 约 **7.4–7.8×**（B200 / H100 NVL）。
- 控制未给视频时可 **当场计算** depth / seg；可用 `mask_path` 做时空二值掩码。
- 2025-12 起支持 Image2Image / ImagePrompt（`docs/inference_image.md`）。
- 后训练：`docs/post-training.md`、`post-training_singleview.md`、`post-training_auto_multiview.md`。

## 对 wiki 的映射

- Transfer 族：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)
- 2.5 论文：[`wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md`](../../wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md)
- Cookbook：[`wiki/entities/cosmos-cookbook.md`](../../wiki/entities/cosmos-cookbook.md)
