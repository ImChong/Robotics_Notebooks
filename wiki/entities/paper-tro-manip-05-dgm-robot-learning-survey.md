---
type: entity
tags: [paper, manipulation, tro-manip-survey, survey, generative-models, diffusion-policy, fair, nvidia]
status: complete
updated: 2026-07-08
arxiv: "2408.04380"
summary: "系统梳理 EBM/扩散/动作值图/GAN 及 VAE→流匹配在 LfD 中的应用、OOD 泛化设计与落地局限。"
related:
  - ../overview/tro-manip-5-papers-technology-map.md
  - ../overview/tro-manip-category-04-generative-models-survey.md
  - ../methods/diffusion-policy.md
  - ../methods/behavior-cloning.md
  - ../concepts/diffusion-model.md
sources:
  - ../../sources/papers/tro_manip_survey_05_dgm_survey.md
  - ../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md
  - ../../sources/papers/tro_manip_5_papers_catalog.md
---

# DGM Robot Learning Survey

**A Survey on Deep Generative Models for Robot Learning From Multimodal Demonstrations** 收录于 [深蓝具身智能 · T-RO 2026 操作学习精选](https://mp.weixin.qq.com/s/nswA-jCGC3kr9iQjhRRuXQ) **第 05/5** 篇，归类为 **04 生成模型综述**。

## 一句话定义

对 **深度生成模型** 在多模态机器人演示学习中的模型类型、应用场景与 **分布外泛化** 设计决策做统一综述，覆盖 EBM、扩散、动作值图、GAN 及 VAE→流匹配演进。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DGM | Deep Generative Model | 深度生成模型 |
| EBM | Energy-Based Model | 能量基模型 |
| GAN | Generative Adversarial Network | 生成对抗网络 |
| OOD | Out-of-Distribution | 分布外泛化 |
| LfD | Learning from Demonstrations | 从演示中学习 |

## 为什么重要

- 生成模型（尤其 **扩散策略**）正重塑机器人模仿学习，但模型族与应用场景分散在多篇论文中。
- 综述同时给出 **OOD 泛化** 的设计选择与 **实时推理 / 多模态对齐 / 安全** 等落地局限，适合作为选型入口。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 05/5 |
| 分组 | 04 生成模型综述 |
| 机构 | FAIR；英伟达（NVIDIA）等 |
| 出处 | IEEE T-RO 2026, vol. 42, pp. 60–79 · arXiv:2408.04380 |
| 论文/项目 | <https://arxiv.org/abs/2408.04380> |

## 核心机制（归纳）

### 1）模型类型轴

- 能量基模型（EBM）、**扩散模型**、动作值映射（Action Value Maps）、GAN。
- 技术演进：VAE → 扩散 → **流匹配（Flow Matching）** 等。

### 2）应用场景轴

- 抓取生成、轨迹生成、代价函数学习等 LfD 子任务。

### 3）泛化与局限

- 回顾提升 **OOD 泛化** 的社区设计决策。
- 指出 **实时推理速度**、多模态输入对齐、安全保障等仍待解决。

## 常见误区

1. 综述是 **选型地图**，不能替代单篇方法（如 Diffusion Policy）的实验协议深读。
2. 「生成模型更好」须结合 **任务延迟预算** 与 **部署安全** 一并判断。

## 实验与评测

- 综述文；以原文表格与引用链为准，无单一 benchmark 分数。

## 与其他页面的关系

- 技术地图：[tro-manip-5-papers-technology-map.md](../overview/tro-manip-5-papers-technology-map.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [Canonical Policy](./paper-tro-manip-02-canonical-policy.md)（生成式策略头实例）

## 参考来源

- [tro_manip_survey_05_dgm_survey.md](../../sources/papers/tro_manip_survey_05_dgm_survey.md)
- [wechat_shenlan_tro_manip_5_papers_survey.md](../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md)
- 论文：<https://arxiv.org/abs/2408.04380>

## 推荐继续阅读

- [Diffusion Model](../concepts/diffusion-model.md)
- [操作 VLA 与视频-动作架构选型](../queries/manipulation-vla-architecture-selection.md)
