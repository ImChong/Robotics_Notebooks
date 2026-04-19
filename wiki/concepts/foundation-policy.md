---
type: concept
tags: [foundation-policy, vla, rt1, rt2, pi0, octo, generalist, pretraining, manipulation]
related:
  - ../methods/imitation-learning.md
  - ../methods/diffusion-policy.md
  - ../methods/policy-optimization.md
  - ../methods/model-based-rl.md
  - ../tasks/loco-manipulation.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/rl_foundation_models.md
  - ../../sources/papers/imitation_learning.md
summary: "Foundation Policy 指在大规模多任务机器人数据上预训练的通用策略模型，是 VLA 和通用操作策略的抽象母类。"
---

# Foundation Policy（基础策略模型）

## 一句话定义

**Foundation Policy（基础策略模型）**：在大规模多任务、多机器人形态演示数据上预训练的通用机器人策略，通过"规模化预训练 + 任务微调"范式，将跨任务泛化能力迁移到新场景——是 NLP 基础模型范式向物理控制的延伸。

## 为什么重要

> "规模化数据 + Transformer 架构 → 跨任务泛化" — RT-1 的核心命题，开创了机器人基础模型方向。

传统机器人策略学习每个任务独立训练，无法复用跨任务知识。基础策略模型试图从根本上解决这一问题：训练一次，泛化到数百乃至数千个任务。

---

## 代表模型

### RT-1（Brohan et al., 2022）
- **数据**：130k+ 真实机器人操作演示，覆盖 700+ 技能
- **架构**：Transformer + EfficientNet 视觉编码器，直接输出 token 化动作
- **意义**：首个在大规模真实数据上证明泛化能力的机器人 Transformer 模型

### RT-2（Brohan et al., 2023）
- **架构**：VLA（Vision-Language-Action）模型；基于 PaLM-E 视觉-语言大模型微调
- **创新**：Web 知识（语言常识、视觉语义）直接迁移到物理控制
- **结果**：泛化能力显著超过 RT-1；可通过自然语言指令驱动低级动作

### π₀（Black et al., 2024）
- **架构**：VLA + Flow Matching 连续动作生成
- **意义**：Physical Intelligence 核心模型；统一了 IL、RL 和 VLA 三类方法
- **优势**：Flow Matching 生成连续动作分布，质量优于 Transformer 直接回归；适合人形操作任务

### Octo（2023）
- **数据**：Open X-Embodiment 800k 演示，跨 22 种机器人形态
- **意义**：第一个开源通用机器人策略；多形态预训练 + fine-tune 范式被广泛采用

### TD-MPC2（Hansen et al., 2024）
- **架构**：隐空间世界模型 + Temporal Difference Learning
- **创新**：单一模型在 80+ 任务上统一训练；model-based RL 在样本效率上的突破

---

## 核心架构范式

```
视觉输入（摄像头）
语言指令（目标描述）
        ↓
  视觉-语言编码器
（CLIP / PaLM-E / Flamingo）
        ↓
  动作解码器
  （Transformer / Diffusion / Flow Matching）
        ↓
  低级关节动作 / 末端位姿
```

**VLA（Vision-Language-Action）** 是当前主流架构：语言作为任务规范，视觉作为状态输入，输出低级机器人控制指令。

---

## 与传统方法的对比

| 维度 | 传统 BC / RL | Foundation Policy |
|------|------------|-------------------|
| 训练数据 | 单任务演示（百~千条） | 多任务大规模（10万~百万条） |
| 泛化方式 | 任务特定 | 零样本 / 少样本迁移 |
| 新任务适应 | 重新训练 | Fine-tune 或 Prompt |
| 计算成本 | 低 | 极高（预训练阶段） |
| 控制精度 | 高（专用策略） | 中（通用策略代价） |

---

## 当前局限

1. **数据规模是瓶颈**：RT-1 需要 130k+ 真实演示；数据采集成本极高，异构性大
2. **Locomotion 基础模型尚未出现**：当前基础模型主要针对操作（pick-place）；跨地形、跨形态的 locomotion 基础模型是下一个前沿
3. **低级控制精度不足**：通用模型在精细操作（螺丝拧紧、接线）上仍弱于专用策略
4. **实时性**：大模型推理延迟与机器人控制高频需求（100Hz+）之间仍有矛盾

---

## 关联页面
- [模仿学习（Imitation Learning）](../methods/imitation-learning.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [VLA](../methods/vla.md)
- [Foundation Policy for Humanoids（Query 实践指南）](../queries/foundation-policy-for-humanoids.md)
- [Policy Optimization](../methods/policy-optimization.md)
- [Model-Based RL](../methods/model-based-rl.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [操作任务（Manipulation）](../tasks/manipulation.md)

## 参考来源
- [rl_foundation_models.md](../../sources/papers/rl_foundation_models.md)
- [imitation_learning.md](../../sources/papers/imitation_learning.md)
- [机器人论文阅读笔记：GR00T N1](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/02_High_Impact_Selection/GR00T_N1_Humanoid_Foundation_Model/GR00T_N1_Humanoid_Foundation_Model.html)
