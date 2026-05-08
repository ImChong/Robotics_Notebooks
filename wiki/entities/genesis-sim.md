---
type: entity
title: Genesis (仿真器)
tags: [simulation, physics-engine, robot-learning, differentiable]
summary: "Genesis 是新兴的高性能多物理场仿真平台，支持刚体、流体及微分仿真，适用于大规模并行机器人数据生成。"
updated: 2026-05-07
---

# Genesis (仿真器)

**Genesis** 是具身智能领域新兴的高性能**物理仿真与数据生成平台**。它通常与 [isaac-gym-isaac-lab](isaac-gym-isaac-lab.md) 并列，作为新一代大规模并行仿真的代表，特别强调在处理柔性体、流体以及多物理场耦合方面的能力。

## 为什么重要？

随着具身智能向复杂环境渗透，传统的刚体仿真已显不足。Genesis 的核心价值在于：
- **万物可仿**：不仅支持刚体（Rigid Body），还支持流体、布料、软体等多种物理实体的统一建模。
- **微分仿真**：支持自动微分（Differentiable Simulation），允许直接对物理过程进行梯度下降优化。
- **极致并行**：充分利用 GPU 并行能力，单机可支持数万个智能体同时训练。

## 核心特性

- **多物理场耦合**：能够处理复杂的交互场景，如机器人切菜、倒水或穿衣。
- **高效率渲染**：内置高性能渲染器，为视觉策略训练提供逼真且高效的合成数据。
- **现代 API**：采用更符合现代 AI 开发者习惯的 Pythonic 接口，降低了环境搭建的复杂度。

## 与其他系统的关系

- **推荐背景**：[xbotics-embodied-guide](../../sources/repos/xbotics-embodied-guide.md) 将 Genesis 推荐为实战路线图中的核心工具之一。
- **对比**：相比 [mujoco](mujoco.md)，Genesis 的并行化程度更高；相比传统的 [sapien](sapien.md)，它在非刚体仿真方面具有显著优势。

## 名称辨析（易混品牌）

英文 **Genesis** 在机器人领域至少对应两条独立线索：本页的 **Genesis-Embodied-AI 开源仿真平台**，以及 **Genesis AI 公司** 与其 **GENE-26.5** 操作基础模型（闭源产品与演示为主）。二者不应在文献或工程选型中混为一谈；参见实体页 [GENE-26.5（Genesis AI）](gene-26-5-genesis-ai.md) 与原始汇编 [genesis_gene_ecosystem](../../sources/papers/genesis_gene_ecosystem.md)。

## 关联页面

- [GS-Playground](./gs-playground.md) — 同为新一代高吞吐仿真，以批量 3DGS 渲染换取光真实感视觉观测（RSS 2026）
- [GENE-26.5（Genesis AI）](gene-26-5-genesis-ai.md) — 与公司品牌相近的机器人基础模型产品线（非本仿真仓库）

## 参考来源

- [Xbotics-Embodied-Guide](../../sources/repos/xbotics-embodied-guide.md)
- [机器人仿真工具核心论文（含 Genesis arXiv 摘录）](../../sources/papers/simulation_tools.md)
- [genesis_gene_ecosystem（Genesis / GENE 资料总档）](../../sources/papers/genesis_gene_ecosystem.md)
- [Genesis Project Page](https://genesis-world.github.io/)
- [Genesis: A Generative and Universal Physics Engine（arXiv）](https://arxiv.org/abs/2412.12919)
