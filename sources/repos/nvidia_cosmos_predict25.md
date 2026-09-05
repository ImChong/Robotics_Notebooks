# nvidia-cosmos/cosmos-predict2.5

> 来源归档

- **标题：** Cosmos-Predict2.5
- **类型：** repo
- **组织：** nvidia-cosmos
- **代码：** <https://github.com/nvidia-cosmos/cosmos-predict2.5>
- **论文：** <https://arxiv.org/abs/2511.00062>
- **Stars：** ~1.4k（2026-09-05）
- **入库日期：** 2026-09-05
- **一句话说明：** Cosmos 2.5 代 **视频世界基础模型** 官方仓：flow matching 统一 Text2World / Image2World / Video2World；配套 Transfer2.5 / Reason1。README 已引导迁移到 Cosmos 3。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md`](../../wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（代码 Apache-2.0；权重 NVIDIA Open Model License） |
| **Predict** | <https://github.com/nvidia-cosmos/cosmos-predict2.5> |
| **Transfer** | <https://github.com/nvidia-cosmos/cosmos-transfer2.5> |
| **Reason** | <https://github.com/nvidia-cosmos/cosmos-reason1> |
| **Cookbook** | <https://github.com/nvidia-cosmos/cosmos-cookbook> |
| **权重** | <https://huggingface.co/nvidia/Cosmos-Predict2.5-2B>、`-14B` |
| **维护** | README **Important**：本仓不再积极开发，仅有限维护；新功能聚焦 [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) |

## README 要点（2026-09-05）

- Cosmos WFM 三类可后训练模型：predict / transfer / reason。
- Predict2.5 用 **Cosmos-Reason1** 作文本编码器，相对 Predict1 提升画质与指令对齐。
- 公开变体：2B/14B pre-trained 与 post-trained；2B distilled；`auto/multiview`（7 摄驾驶）；`robot/action-cond`；`robot/multiview-agibot`；`robot/policy`（Libero / RoboCasa）。
- 推理与后训练文档在 `docs/inference*.md`、`docs/post-training*.md`；Diffusers：`Cosmos2_5_PredictBasePipeline`。
- 2026-02-23 仍发布过 Action-Cond 蒸馏指南与 Policy 权重；其后官方叙事转向 Cosmos 3。

## 对 wiki 的映射

- 论文实体（本 arXiv 唯一节点）：[`wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md`](../../wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md)
- 平台总览：[`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)
- 下一代：[`wiki/entities/cosmos-3.md`](../../wiki/entities/cosmos-3.md)
- Transfer：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)、[`sources/repos/nvidia_cosmos_transfer25.md`](./nvidia_cosmos_transfer25.md)
- Cookbook：[`wiki/entities/cosmos-cookbook.md`](../../wiki/entities/cosmos-cookbook.md)
