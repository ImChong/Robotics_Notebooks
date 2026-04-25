---
type: method
tags: [vla, foundation-policy, embodied-ai, qwen, benchmark]
status: complete
updated: 2026-04-25
related:
  - ./vla.md
  - ../concepts/foundation-policy.md
  - ../entities/nvidia-omniverse.md
  - ./diffusion-policy.md
summary: "StarVLA 是具身智能领域的极简基准模型，证明了强 VLM 底座（如 Qwen3-VL）配合简单 MLP 动作头即可实现最先进的控制性能。"
---

# StarVLA

**StarVLA**（尤其是其首个技术报告版本 **StarVLA-$\alpha$**）是一个旨在降低 Vision-Language-Action (VLA) 系统复杂性的开源基准模型与框架。

## 一句话定义

> StarVLA 证明了：不需要复杂的动作分词或扩散生成头，只需“强大的 VLM 底座 + 简单的 MLP 回归头”，就能在 LIBERO 和 SimplerEnv 等基准测试中打破 SOTA。

---

## 核心设计理念：极简主义

在 OpenVLA 和 RT-2 之后，具身智能社区倾向于设计越来越复杂的动作表达方式。StarVLA 逆流而上，提出了两个核心观察：

1.  **VLM 能力决定上限**：随着视觉语言模型（如 Qwen3-VL）的爆炸式进化，底座模型已经具备了极强的空间推理和指令理解能力。
2.  **动作头应保持轻量**：在强大的特征提取能力下，一个简单的 MLP（多层感知机）足以完成从隐空间到连续控制量（Action）的映射。

## 技术规格

-   **底座模型**：Qwen3-VL (4B 为平衡点，支持 0.8B 至 9B)。
-   **动作专家 (Action Head)**：
    -   **OFT (Default)**：基于 MLP 的连续动作回归。
    -   **FAST**：离散 Token 预测。
    -   **PI**：基于 Flow-matching 的生成式动作头。
-   **代码架构**：积木式（Lego-like）模块化设计，支持子模块独立烟雾测试（Smoke Test）。

## 性能表现

| 基准测试 | StarVLA-$\alpha$ 成功率 | 对比模型 |
| :--- | :--- | :--- |
| **LIBERO** | **98.8%** | 超越 OpenVLA |
| **SimplerEnv** | **76.0%** | OpenVLA (34.3%) |
| **真机 (ARX5)** | **比 $\pi_{0.5}$ 高 20%** | - |

### 零样本泛化 (Zero-shot)
StarVLA 在面对环境光照变化、物体位姿偏移以及从未见过的背景（OOD 场景）时，表现出了极强的鲁棒性，这得益于 Qwen3-VL 强大的视觉编码能力。

---

## 主要方法路线

-   **理论基础**：[基础策略 (Foundation Policy)](../concepts/foundation-policy.md) / [VLA 形式化](../formalizations/vla-tokenization.md)
-   **模型底座**：Qwen3-VL / Florence-2
-   **动作生成**：MLP Regression (OFT) / Flow-matching (PI)
-   **训练数据**：Open X-Embodiment (OXE) / 仿真数据合成

---

## 为什么选择 StarVLA？

1.  **作为科研 Baseline**：它的架构极简，非常适合作为新研究的起始对比点。
2.  **作为工业部署**：由于动作头极轻且支持中等规模 VLM 底座，它在实时性要求高的机器人任务中具有优势。
3.  **开发体验**：其高度模块化的代码库极大降低了具身大模型的上手门槛。

## 参考来源

-   Ye et al., *StarVLA-α: Reducing Complexity in Vision-Language-Action Systems* (2026) — 原论文
-   [sources/papers/star_vla.md](../../sources/papers/star_vla.md) — 论文 ingest 档案
-   [sources/repos/star_vla.md](../../sources/repos/star_vla.md) — GitHub 仓库 ingest 档案

## 关联页面

-   [VLA (Vision-Language-Action)](./vla.md) — 具身大模型全景
-   [Foundation Policy](../concepts/foundation-policy.md) — 基础策略模型概念
-   [Diffusion Policy](./diffusion-policy.md) — 另一种主流的生成式动作头路线
