---
type: entity
tags: [google-deepmind, vla, gemini, embodied-ai, product, hmi-papers]
title: Gemini Robotics
status: complete
summary: "Gemini Robotics 是 Google DeepMind 基于 Gemini 多模态栈发布的机器人视觉–语言–动作与具身推理模型族（含 ER / 1.5 等迭代），强调泛化、交互与自然语言指令。"
updated: 2026-08-04
related:
  - ../methods/vla.md
  - ../methods/robotics-transformer-rt-series.md
  - ./paper-palm-e-embodied-language-model.md
  - ./perceptron-egocentric.md
  - ../queries/hmi-papers-coverage.md
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Gemini Robotics

**Gemini Robotics** 是 Google DeepMind 面向物理交互的 Gemini 系列机器人模型族。HMI 论文/报告总索引将其收录为 **P061**（世界模型、VLA 与 Agent）。

## 一句话定义

**Gemini Robotics**：面向物理交互的 Gemini 系列机器人模型，通常包含 **VLA 式策略骨干**与强调空间 / 任务推理的 **Embodied Reasoning（ER）** 变体；后续迭代（如公开资料中的 1.5）在长程任务分解与跨本体动作迁移等方向扩展能力叙事。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略 |
| ER | Embodied Reasoning | 强调空间/任务推理的变体 |
| VLM | Vision-Language Model | Gemini 多模态语言主干 |
| Sim2Real | Simulation to Real | 真机部署时的迁移议题 |

## 为什么重要

- 代表闭源多模态大模型接入机器人控制的产品化叙事，常与开源 [Octo](../methods/octo-model.md) / [OpenVLA](./openvla.md) 对照阅读。
- ER 变体把「会做动作」与「会分解长程任务」拆开讨论，避免把演示视频当成可复现基线。
- 在自动标注与评测生态中常被用作强教师/对照（见 Perceptron Egocentric）。

## 核心原理

公开材料通常强调：多模态理解 → 任务/子任务推理 → 动作或技能调用。与早期 [PaLM-E](./paper-palm-e-embodied-language-model.md) 一脉相承的是「传感器与语言共享推理上下文」，但 Gemini Robotics 更明确地把输出接到机器人执行接口，并按版本迭代长程与跨本体能力。

```mermaid
flowchart LR
  A["视觉 / 语言指令"] --> B["Gemini Robotics"]
  B --> C["ER 子任务 / 空间推理"]
  C --> D["动作或技能接口"]
  D --> E["低层控制器 / 真机"]
```

## 工程实践

1. 能力边界与数值以 **官方博客 + 技术报告 PDF** 为准，不要只引用二手摘要。
2. 与开源通用策略对比时，对齐任务定义、数据可见性与是否允许专用微调。
3. **自动标注对照：** Macrodata **WGO-Bench** 以 **Gemini 3.5 Flash + Gemini Robotics ER-1.6** 构建机器人子任务分段管线——见 [Perceptron Egocentric](./perceptron-egocentric.md)。

| 检查项 | 建议 |
|--------|------|
| 开源 | **未开源权重**（产品/研究报告形态） |
| 评测 | 分清演示、受控实验与可复现基准 |
| 人形 | 全身平衡与接触仍通常外挂低层栈 |

## 源码运行时序图

**不适用**（无公开可运行训练/推理仓库作为本库复现入口）。

## 结论

**Gemini Robotics 适合作为「闭源多模态机器人栈」对照节点，不适合当作可本地复现的开源基线。**

- 先读官方技术报告再引用指标。
- 与 PaLM-E / RT / 开源 VLA 对照时分开「推理」与「可部署动作接口」。
- 自动标注场景可作强教师，但仍需本库评测页交叉验证。
- 人形全身控制不要假设已被 VLA 层替代。

## 局限与风险

- 权重与完整训练数据不可复现。
- 产品迭代快，博客数字需标注读取日期。
- 勿把营销演示直接写成学术 SOTA。

## 关联页面

- [VLA](../methods/vla.md)
- [Foundation Policy](../concepts/foundation-policy.md)
- [PaLM-E](./paper-palm-e-embodied-language-model.md)
- [Perceptron Egocentric](./perceptron-egocentric.md)
- [HMI 论文导读](../queries/hmi-papers-coverage.md)
- [CLIFT](./paper-clift-closed-loop-iterative-finetuning.md) — 通过托管 SFT API 把 Gemini Robotics On-Device 适配成人形专才（arXiv:2607.29172）

## 参考来源

- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
- [humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)

## 推荐继续阅读

- [Gemini Robotics 博客](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/)
- [Gemini Robotics 1.5](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)
- [技术报告 PDF](https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-1.5-Tech-Report.pdf)
