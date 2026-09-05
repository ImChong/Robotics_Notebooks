# nvidia-cosmos/cosmos-transfer1

> 来源归档

- **标题：** Cosmos-Transfer1
- **类型：** repo
- **组织：** nvidia-cosmos
- **代码：** <https://github.com/nvidia-cosmos/cosmos-transfer1>
- **论文：** <https://arxiv.org/abs/2503.14492>
- **项目页：** <https://research.nvidia.com/labs/cosmos-lab/cosmos-transfer1/>
- **权重集合：** <https://huggingface.co/collections/nvidia/cosmos-transfer1-67c9d328196453be6e568d3e>
- **Stars：** ~820（2026-09-05）
- **入库日期：** 2026-09-05
- **一句话说明：** Cosmos 1.0 代 **world-to-world transfer** 官方仓：单模态 ControlNet、自适应 MultiControlNet、4K 上采样、AV Sample、蒸馏 Edge，以及后训练 / 预训练脚本。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-cosmos-transfer1.md`](../../wiki/entities/paper-cosmos-transfer1.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（代码 Apache-2.0；权重 NVIDIA Open Model License） |
| **代码** | <https://github.com/nvidia-cosmos/cosmos-transfer1> |
| **权重** | Cosmos-Transfer1-7B、Sample-AV、4KUpscaler、Edge Distilled、Single2MultiView |
| **维护** | README **Important**：有限维护；新产品走 [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) |

## README 要点（2026-09-05）

- 单模态：seg / depth / edge / blur / LiDAR / HDMap / Keypoint（1.1）。
- 首选模式：MultiControlNet + 时空控制图（Transfer1-7B）。
- 推理示例在 `examples/inference_cosmos_transfer1_7b*.md`；后训练 `examples/training_cosmos_transfer_7b.md`；蒸馏 `examples/distillation_cosmos_transfer1_7b.md`。
- 2025-08：Edge Distilled，**1 步** vs 原 36 步。
- 2025-05：AV Single2MultiView。
- 机器人增广工作流：`cosmos_transfer1/auxiliary/robot_augmentation/`。
- 内容过滤用 Llama Guard 3（独立许可）。

## 对 wiki 的映射

- 论文：[`wiki/entities/paper-cosmos-transfer1.md`](../../wiki/entities/paper-cosmos-transfer1.md)
- Transfer 族：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)
- 项目页：[`sources/sites/cosmos-transfer1-project.md`](../sites/cosmos-transfer1-project.md)
