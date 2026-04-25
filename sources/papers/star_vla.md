# StarVLA-alpha

> 来源归档（ingest）

- **标题：** StarVLA-α: Reducing Complexity in Vision-Language-Action Systems
- **类型：** paper
- **来源：** arXiv:2604.11757
- **入库日期：** 2026-04-25
- **最后更新：** 2026-04-25
- **一句话说明：** 提出了一种基于 Qwen3-VL 的极简 VLA 基准模型，证明了强 VLM 底座配合简单 MLP 动作头即可在多项基准测试中超越复杂模型。

## 核心内容摘要

### 1. 设计哲学
- **Minimalist VLA (极简主义)**：摒弃了复杂的动作分词（tokenization）或复杂的生成式动作头（如扩散模型），回归到“强 VLM + 简单 MLP 回归”的方案。
- **Modular Codebase (模块化)**：采用“积木式”设计，低耦合、高内聚，方便更换 VLM 底座（Qwen3, Florence-2 等）和动作专家（FAST, OFT, PI, GR00T）。

### 2. 技术架构
- **底座 (Backbone)**：主打 **Qwen3-VL (4B)**，平衡了性能与显存效率。
- **动作头 (Action Head)**：默认使用 **MLP Head (OFT 路线)**，读取 VLM 的隐藏层状态直接回归连续动作。
- **训练范式**：支持多基准联合训练（Generalist co-training），涵盖 LIBERO, SimplerEnv, RoboTwin, RoboCasa 等。

### 3. 性能表现
- **LIBERO**：平均成功率 **98.8%**。
- **SimplerEnv (Google VM)**：成功率 **76.0%**（远超 OpenVLA 的 34.3%）。
- **真机测试**：在 ARX5 机器人上比 $\pi_{0.5}$ 成功率高出 20%。
- **鲁棒性**：对环境扰动（LIBERO-Plus）和未见任务（OOD）表现出极强的零样本迁移能力。

## 关键术语
- **Qwen3-VL**: 阿里巴巴开源的最新一代多模态大模型底座。
- **OFT-style Head**: 连续动作回归头。
- **Generalist VLA**: 能够同时处理多个基准测试和机器人形态的通用策略。

## 关联 Wiki 页面
- [VLA (Vision-Language-Action)](../../wiki/methods/vla.md)
- [Foundation Policy](../../wiki/concepts/foundation-policy.md)
- [StarVLA (Method)](../../wiki/methods/star-vla.md)

## 当前提炼状态
- [x] 核心架构与设计哲学
- [x] 性能基准对比
- [ ] 后续：深入研究其模块化代码实现中的数据加载 pipeline
