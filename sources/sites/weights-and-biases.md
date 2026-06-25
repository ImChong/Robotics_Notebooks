# Weights & Biases（W&B）官方产品站

> 来源归档

- **标题：** Weights & Biases — The AI developer platform
- **类型：** site（官方产品站）
- **来源：** Weights & Biases, Inc.
- **链接：** https://wandb.ai/site/
- **入库日期：** 2026-06-25
- **一句话说明：** 面向 AI 研发团队的实验追踪、模型/数据制品托管、Agent 可观测性与协作平台；机器人 RL 仓库中常与 PyTorch / Lightning / Hugging Face 集成，用于云端对比 run、共享 checkpoint 与视频日志。

## 为什么值得保留

- **训练闭环标配**：本库大量 RL / 模仿学习仓库（mjlab、Holosoma、MimicKit、legged_gym 生态 fork 等）支持 `wandb` logger，用于多机实验对比与 artifact 分发。
- **超越标量曲线**：除 scalar 外支持 **Artifacts**（模型权重、ONNX）、**Reports**、训练视频与超参 sweep，适合团队协作而非单机 debug。
- **与 TensorBoard 互补**：许多框架同时支持 `tb` 与 `wandb`；选型对比见 [wandb-vs-tensorboard](../../wiki/comparisons/wandb-vs-tensorboard.md)。

## 平台模块（官网 2026-06 摘录）

| 模块 | 路径/能力 | 说明 |
|------|-----------|------|
| **Models** | `/site/models` | 经典 **实验追踪**：`wandb.init` → `run.config` / `run.log` / `run.watch`；PyTorch、TF/Keras、HF Transformers、Lightning 等一键集成 |
| **Weave** | `/site/weave` | **Agent / LLM 应用** 追踪：`@weave.op()` 记录调用链、检索与多步推理（与本库机器人主线弱相关，但同属 W&B 生态） |
| **Training** | `/site/wb-training` | 托管微调与 serverless RL（企业向） |
| **Inference** | `/site/inference` | 托管推理与模型探索 |
| **Registry** | `/site/registry` | 数据集 / 模型 / Prompt / 代码 / 元数据版本注册表 |
| **Core** | Reports、Automations、SDK、Agent Skills / MCP | 报告、自动化、开放 SDK |
| **部署** | SaaS / Dedicated / Customer-managed | ISO 27001、SOC 2、HIPAA 等合规选项 |

## SDK 最小用法（官网示例摘要）

```python
import wandb

run = wandb.init(project="my-model-training-project")
run.config = {"epochs": 1337, "learning_rate": 3e-4}
run.log({"metric": 42})
run.log_artifact("./my_model.pt", type="model")
```

常见集成：

- **PyTorch Lightning**：`WandbLogger(project=...)`
- **Hugging Face Trainer**：`TrainingArguments(..., report_to="wandb")`
- **从 run 拉 checkpoint**：`wandb_run_path` → 评估脚本（如 axellwppr motion_tracking、SMP G1）

## 与本仓库现有资料的关系

| 资料 | 关系 |
|------|------|
| [mjlab.md](../repos/mjlab.md) | 内置 W&B 实验追踪 |
| [holosoma.md](../repos/holosoma.md) | 视频日志、ONNX 自动上传、从 Wandb 加载 checkpoint |
| [amp_mjlab.md](../repos/amp_mjlab.md) | 以 TensorBoard 为主；可与 W&B 对照 |
| [mimickit.md](../repos/mimickit.md) | `logger_type` 支持 `"tb"` 与 `"wandb"` |
| [robot-policy-debug-playbook](../../wiki/queries/robot-policy-debug-playbook.md) | 训练期监控工具链 |

## 对 wiki 的映射

- 升格实体：[weights-and-biases.md](../../wiki/entities/weights-and-biases.md)
- 选型对比：[wandb-vs-tensorboard.md](../../wiki/comparisons/wandb-vs-tensorboard.md)
