---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2512.17900"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_magnet.md
summary: "把 Diffusion Forcing（每个 token 独立加噪/独立去噪的自回归扩散）从单序列搬到多人交互——把每个人的姿态先用 VQ-VAE 压成 token，再把所有人的 token 交错喂给同一个 Transformer，训练时每个 token 独立采噪声、推理时按需控制每个人/每个时刻的噪声等级，从而一个模型同时支持双人/三人/N 人预测、Partner Inpainting、Partner Prediction、超长动作生成等任务。"
---

# MAGNet

**MAGNet: Diffusion Forcing for Multi-Agent Interaction Sequence Modeling** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

把 Diffusion Forcing（每个 token 独立加噪/独立去噪的自回归扩散）从单序列搬到多人交互——把每个人的姿态先用 VQ-VAE 压成 token，再把所有人的 token 交错喂给同一个 Transformer，训练时每个 token 独立采噪声、推理时按需控制每个人/每个时刻的噪声等级，从而一个模型同时支持双人/三人/N 人预测、Partner Inpainting、Partner Prediction、超长动作生成等任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/MAGNet__Diffusion_Forcing_for_Multi-Agent_Interaction_Sequence_Modeling/MAGNet__Diffusion_Forcing_for_Multi-Agent_Interaction_Sequence_Modeling.html> |
| arXiv | <https://arxiv.org/abs/2512.17900> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_magnet.md](../../sources/papers/humanoid_pnb_magnet.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/MAGNet__Diffusion_Forcing_for_Multi-Agent_Interaction_Sequence_Modeling/MAGNet__Diffusion_Forcing_for_Multi-Agent_Interaction_Sequence_Modeling.html>
- 论文：<https://arxiv.org/abs/2512.17900>

## 推荐继续阅读

- [机器人论文阅读笔记：MAGNet](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/MAGNet__Diffusion_Forcing_for_Multi-Agent_Interaction_Sequence_Modeling/MAGNet__Diffusion_Forcing_for_Multi-Agent_Interaction_Sequence_Modeling.html)
