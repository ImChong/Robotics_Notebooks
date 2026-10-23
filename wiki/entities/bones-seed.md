---
type: entity
project_id: bones-seed
project: https://bones.studio/datasets/seed
dataset: https://huggingface.co/datasets/bones-studio/seed
tags: [dataset, humanoid, mocap, whole-body-control, human-motion, language-conditioned]
status: complete
updated: 2026-10-23
related:
  - ../comparisons/humanoid-reference-motion-datasets.md
  - ./lafan1-dataset.md
  - ./dataset-bfm-lafan.md
  - ../concepts/behavior-foundation-model.md
sources:
  - ../../sources/datasets/bones-seed.md
  - ../../sources/sites/bones-studio-seed.md
summary: "BONES-SEED 是 Bones Studio 的人形研究动作捕捉数据集：约 142,220 条片段、288 小时，带语言与时序标注，提供 SOMA 与 G1 兼容表示；Hugging Face 访问需接受许可并申请。"
---

# BONES-SEED：日常具身动作数据集

**一句话定义：** BONES-SEED 是面向人形机器人研究的动作捕捉库，提供日常行为、全身动作、语言描述与 SOMA / Unitree G1 兼容轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BONES-SEED | Skeletal Everyday Embodiment Dataset | 人形具身研究的带注释骨架动作数据集 |
| MoCap | Motion Capture | 记录身体运动并生成骨架序列的动作捕捉 |
| SOMA | Unifying Parametric Human Body Models | 用统一人体拓扑与骨架规范承接不同参数化人体模型的框架 |
| G1 | Unitree G1 Humanoid | 数据集提供兼容 G1 MuJoCo 的关节轨迹 |
| BVH | Biovision Hierarchy | SOMA 动作序列使用的骨架层级与运动格式 |

## 为什么重要

BONES-SEED 组合了动作捕捉、自然语言描述、时间分段和机器人格式轨迹，可支持动作检索、语言条件生成、模仿学习与策略训练。相较 [LaFAN1](./lafan1-dataset.md) 的小规模 BVH 基准，它强调动作规模、语义标注和机器人兼容表示；训练价值仍取决于动力学可行性与任务定义。

## 数据集速查

| 项 | 官方数据卡口径 |
|----|----------------|
| 动作数量 | 142,220（71,132 原始 + 71,088 镜像） |
| 总时长 | 约 288 小时（120 fps） |
| 表演者 | 522 人，年龄 17–71 岁 |
| 动作类别 | 8 大类、20 细类，含行走、交流、物体交互、舞蹈、日常活动、运动和特技 |
| 动作表示 | SOMA Uniform、SOMA Proportional、Unitree G1 MuJoCo 兼容 CSV |
| 标注 | 每段最多 6 条自然语言描述、时间分段及骨架元数据 |

## 工程实践

- 先用 Parquet / CSV 元数据按类别、描述和表演者属性筛选，再下载动作。
- Uniform 适合跨片段比较，Proportional 保留表演者比例，G1 CSV 是已重定向关节轨迹；训练前检查关节顺序、限位、速度和接触。
- Hugging Face 要求申请者提交姓名、机构和用途等信息并确认许可资格。使用前阅读[许可全文](https://bones.studio/info/seed-license)并核对当前数据卡。
- 数据许可不自动决定模型、衍生动作及商业使用的许可边界。

## 局限与风险

- 高质量动捕不自动等于物理可执行的机器人轨迹，仍需验证重定向、接触、平衡和执行器能力。
- 数据含表演者动作及元数据，使用需遵循许可和隐私约束。
- HF 页面可见不代表无条件下载或再分发。

## 关联页面

- [人形参考运动与操作数据集选型](../comparisons/humanoid-reference-motion-datasets.md)
- [LaFAN1 数据集](./lafan1-dataset.md)
- [BFM LAFAN 数据索引](./dataset-bfm-lafan.md)
- [行为基础模型](../concepts/behavior-foundation-model.md)

## 参考来源

- [BONES-SEED 数据集归档](../../sources/datasets/bones-seed.md)
- [Bones Studio 官方项目页](../../sources/sites/bones-studio-seed.md)
- [Hugging Face 数据卡](https://huggingface.co/datasets/bones-studio/seed)
- [NVIDIA SOMA-X 官方仓库](https://github.com/NVlabs/SOMA-X)

## 推荐继续阅读

- [BONES-SEED 官方数据页](https://bones.studio/datasets/seed)
- [Hugging Face 数据集](https://huggingface.co/datasets/bones-studio/seed)
- [BONES-SEED 许可全文](https://bones.studio/info/seed-license)
- [NVIDIA SOMA-X](https://github.com/NVlabs/SOMA-X)
