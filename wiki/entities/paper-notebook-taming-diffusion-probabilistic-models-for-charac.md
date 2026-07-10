---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2404.15121"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_taming-diffusion-probabilistic-models-for-charac.md
summary: "本文提出一个新颖的角色控制框架，有效利用动作扩散概率模型生成高质量、多样的角色动画，并实时响应各种动态用户控制信号。系统核心是一个基于 Transformer 的条件自回归动作扩散模型（Conditional Autoregressive Motion Diffusion Model, CAMDM），依据历史动作与粗粒度用户控制生成可能的未来动作。为实现高质量实时控制，框架用三招：条件单独 token 化、对历史动作的无分类器引导（classifier-free guidance）、以及启发式未来轨迹外延（提升计算效率）。这是首个支持基于用户交互控制实时生成高质量多样角色动画的模型，单一统一模型支持多种动画风格与多样行走技能。SIGGRAPH 2024。"
---

# Taming Diffusion Probabilistic Models for Character Control

**Taming Diffusion Probabilistic Models for Character Control** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

本文提出一个新颖的角色控制框架，有效利用动作扩散概率模型生成高质量、多样的角色动画，并实时响应各种动态用户控制信号。系统核心是一个基于 Transformer 的条件自回归动作扩散模型（Conditional Autoregressive Motion Diffusion Model, CAMDM），依据历史动作与粗粒度用户控制生成可能的未来动作。为实现高质量实时控制，框架用三招：条件单独 token 化、对历史动作的无分类器引导（classifier-free guidance）、以及启发式未来轨迹外延（提升计算效率）。这是首个支持基于用户交互控制实时生成高质量多样角色动画的模型，单一统一模型支持多种动画风格与多样行走技能。SIGGRAPH 2024。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| CAMDM | 条件自回归动作扩散模型 |
| Autoregressive | 自回归，依历史生成未来 |
| Classifier-free Guidance | 无分类器引导 |
| Condition Tokenization | 条件单独 token 化 |
| Real-time Control | 实时交互控制 |
| Locomotion Skills | 行走技能 |

## 为什么重要

- **自回归扩散 + 历史条件**让生成式模型可实时响应控制，与人形在线控制需求一致；
- **无分类器引导用在"历史动作"上**是个巧思，可借鉴到动作跟踪；
- **启发式轨迹外延**降算力，是把扩散压进实时的实用技巧；
- 角色控制的实时扩散经验可迁移到人形 loco 控制（与 Heracles/SafeFlow 的生成式控制呼应）。

## 解决什么问题

扩散模型质量高但**难实时交互控制**： - 采样慢、难响应**动态用户控制**； - 要**单一模型多风格**、保**多样性**。

论文要：**驯服**扩散模型，实现**实时、可控、多样**的角色动画生成。

## 核心机制

1. **CAMDM**：Transformer 条件自回归动作扩散；
2. **实时可控**：条件 token 化 + 无分类器引导 + 轨迹外延；
3. **首个实时交互角色动画扩散**：质量与多样兼得；
4. **单一模型多风格 + 多样行走**（SIGGRAPH 2024）。

方法拆解（深读笔记小节）：CAMDM：条件自回归动作扩散；三招保质量与实时；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Taming_Diffusion_Probabilistic_Models_for_Character_Control/Taming_Diffusion_Probabilistic_Models_for_Character_Control.html> |
| arXiv | <https://arxiv.org/abs/2404.15121> |
| 作者 | Rui Chen、Mingyi Shi、Shaoli Huang、Ping Tan、Taku Komura、Xuelin Chen |
| 发表 | 2024 年 4 月 |
| 会议 | SIGGRAPH 2024 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_taming-diffusion-probabilistic-models-for-charac.md](../../sources/papers/humanoid_pnb_taming-diffusion-probabilistic-models-for-charac.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Taming_Diffusion_Probabilistic_Models_for_Character_Control/Taming_Diffusion_Probabilistic_Models_for_Character_Control.html>
- 论文：<https://arxiv.org/abs/2404.15121>

## 推荐继续阅读

- [机器人论文阅读笔记：Taming Diffusion Probabilistic Models for Character Control](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Taming_Diffusion_Probabilistic_Models_for_Character_Control/Taming_Diffusion_Probabilistic_Models_for_Character_Control.html)
