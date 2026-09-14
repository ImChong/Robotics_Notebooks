---
type: entity
tags: [paper, humanoid, world-model, mamba, teacher-student, ntu, baai, pku, nju]
status: complete
updated: 2026-09-14
arxiv: "2609.07096"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/generative-world-models.md
  - ./paper-wm-loco.md
sources:
  - ../../sources/papers/robodreamer_arxiv_2609_07096.md
summary: "RoboDreamer（arXiv:2609.07096）：two-stage Teacher-Student; random masking; next-obs consistency; Mamba long sequence; action refinement at inference；截至入库日未见官方代码。"
---

# RoboDreamer（arXiv:2609.07096）

**RoboDreamer**（*RoboDreamer: Anticipatory Humanoid Locomotion with Predictive State-Space Models*，[arXiv:2609.07096](https://arxiv.org/abs/2609.07096)）由 **南洋理工大学（NTU）；北京智源（BAAI）；北京大学（PKU）；南京大学（NJU）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)）。

## 一句话定义

RoboDreamer：基于预测状态空间模型的前瞻式人形运动控制 — two-stage Teacher-Student。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PSSM | Predictive State-Space Model | 预测状态空间模型 |
| Mamba | Mamba SSM | 长序列状态空间骨干 |
| TS | Teacher-Student | 两阶段蒸馏 |

## 为什么重要

人形行走需要短 horizon 预测支撑落脚与平衡；RSSM 类模型在长序列与推理细化上仍有缺口。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 南洋理工大学（NTU）；北京智源（BAAI）；北京大学（PKU）；南京大学（NJU） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

Teacher 用 privileged 信息训练 Mamba-based PSSM，优化 next-observation consistency；Student 在随机掩码下学习；推理阶段对动作做 refinement。

### 流程总览

```mermaid
flowchart LR
  priv[特权观测] --> pssm[Mamba PSSM Teacher]
  pssm --> mask[随机掩码 Student]
  mask --> policy[部署策略]
  policy --> refine[推理动作细化]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-14** arXiv 与常见项目页 **未见** 官方可运行代码仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 掩码比例与序列长度影响前瞻质量；推理 refinement 步数需权衡延迟。 |

## 实验与评测

仿真与真机 anticipatory locomotion；长序列预测误差与跟踪稳定性。

## 结论

RoboDreamer 用 Mamba PSSM 与推理细化把人形行走从反应式推向前瞻式。

1. 两阶段 TS 分离世界模型与部署策略。
2. 随机掩码提升缺失感知鲁棒性。
3. next-obs consistency 约束预测可用性。
4. Mamba 承担长时序。
5. 推理 action refinement 改善落脚时机。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| WM-LOCO | RSSM 共训 PPO，无 Mamba 长序列 |
| 纯 PPO 行走 | 无显式前瞻预测 |

## 局限与风险

推理细化增加算力；极端感知失效时 PSSM 误差仍累积。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [generative-world-models](../methods/generative-world-models.md)
- [./paper-wm-loco.md](./paper-wm-loco.md)

## 参考来源

- [robodreamer_arxiv_2609_07096.md](../../sources/papers/robodreamer_arxiv_2609_07096.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.07096](https://arxiv.org/abs/2609.07096)
