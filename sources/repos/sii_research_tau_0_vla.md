# sii-research/tau-0-vla

> 来源归档

- **标题：** τ₀-VLA（官方实现）
- **类型：** repo
- **组织：** sii-research（上海创智学院等，以 upstream 为准）
- **代码：** <https://github.com/sii-research/tau-0-vla>
- **项目页：** <https://tau0-vla.github.io/>
- **论文：** <https://arxiv.org/abs/2608.16885>
- **权重：** <https://huggingface.co/sii-research/tau-0-vla>（Apache-2.0）
- **入库日期：** 2026-08-19
- **一句话说明：** τ₀-VLA 官方仓库：**低层 generalist VLA** 后训练与 `deploy.server` 开环评测、LeRobot v3 范例数据；**高层 policy（TTC / 记忆）组件据 README [2026.08.19] 将逐步发布**。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [τ₀-VLA](../../wiki/entities/paper-tau0-vla.md) | 实体归纳页：分层 TTC、记忆、低层 MoT flow VLA |
| [τ₀-World Model](../../wiki/entities/tau0-world-model.md) | 同 Agibot / SII 生态：**测试时想象** 对照——τ₀-WM 在 **action chunk + 视频扩散** 级 propose–evaluate–revise；τ₀-VLA 在 **开放语言子任务** 级 beam search |
| [π₀.₅](../../wiki/entities/paper-pi05-open-world-vla.md) | 长程真机对照基线之一 |
| [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) | 异构小时 generalist VLA 对照 |
| [VLA](../../wiki/methods/vla.md) | 分层子任务接口 + 测试时算力扩展 |

## 部署要点（README 摘要，以克隆时 upstream 为准）

- **环境：** Python 3.11、CUDA 12.8、PyTorch 2.7.1；`bash scripts/setup.sh`。
- **后训练：** `bash scripts/train.sh configs/example_agibot_world_gong/train.yaml --model_name_or_path /path/to/tau-0-vla-checkpoint`；模板见 `configs/_template/` 与 `src/tau0_vla/adapters/_template/`。
- **Serving（v1）：** 仅 **joint-control** checkpoint；`python -m deploy.server --model outputs/<run_name>`；开环 `deploy/openloop.py`。
- **数据：** `example_data/` 为 LeRobot v3 格式 AgiBot World 子集。
- **布局：** `src/tau0_vla/{adapters,data,models,trainer,vlm,utils}`；`deploy/` policy server；文档见 `DATASET_FORMAT.md`、`adapters/README.md`。

## 开源边界（截至 2026-08-19）

| 组件 | 状态 |
|------|------|
| 低层 VLA 权重（HF） | **已发布** |
| 后训练 / deploy / 范例数据 | **已发布** |
| 高层 policy + TTC 栈 | **待逐步发布**（README News 2026.08.19） |
| EEF serving | **本版不支持**（可用 EEF 数据训练，serving 仅 joint-control） |

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/paper-tau0-vla.md`](../../wiki/entities/paper-tau0-vla.md)**；论文摘录见 [`sources/papers/tau0_vla_arxiv_2608_16885.md`](../papers/tau0_vla_arxiv_2608_16885.md)。
