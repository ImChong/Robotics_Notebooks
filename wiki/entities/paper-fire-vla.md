---
type: entity
tags:
  - paper
  - vla
status: complete
updated: 2026-09-26
arxiv: "2608.13395"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md
sources:
  - ../../sources/papers/fire-vla_arxiv_2608_13395.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md
summary: "FIRE-VLA（arXiv:2608.13395）：组内轨迹都很差时 GRPO 难找正确方向；挑出相似失败，用冻结旧模型作 teacher 沿新模型输出继续纠正，配合常规范 RL，减少严重出错与反复失败。…"
---

# FIRE-VLA（arXiv:2608.13395）

**FIRE-VLA**（[arXiv:2608.13395](https://arxiv.org/abs/2608.13395)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第三篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) **训练范式/智驾** 段。

## 一句话定义

**组内轨迹都很差时 GRPO 难找正确方向；挑出相似失败，用冻结旧模型作 teacher 沿新模型输出继续纠正，配合常规范 RL，减少严重出错与反复失败。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| GRPO | Group Relative Policy Optimization | 组内相对策略优化（FIRE-VLA / Temporal GRPO 语境） |
| RL | Reinforcement Learning | 强化学习后训练 |

## 为什么重要

- 组内轨迹都很差时 GRPO 难找正确方向；挑出相似失败，用冻结旧模型作 teacher 沿新模型输出继续纠正，配合常规范 RL，减少严重出错与反复失败。
- 策展机构：哈尔滨工业大学
- 与 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md) 同批工作可横向对照。

## 流程总览

```mermaid
flowchart TD
    A["驾驶提示 x_i = 前视图 + 自车历史描述<br/>当前策略 π_k 采样 G 条 on-policy rollout"] --> B["阶段 1：失败感知路由<br/>组内奖励均值、标准差、有效比例"]
    B -->|"均值与标准差均 ≤ batch 30 分位且有效比例 ≥ 0.5"| C["未解决失败组：GRPO + PSD"]
    B -->|"其余"| D["普通组：仅 GRPO"]
    C --> E["阶段 2：构造两个上下文<br/>学生：x + 学生前缀；教师：同前缀 + 隐藏未来轨迹 τ*"]
    E --> F["阶段 3：特权同策略教学<br/>冻结教师与学生同一轮同权重，仅答案 token 上做 top-16 + 残差尾 JSD，梯度只经学生"]
    F --> G["阶段 4：策略更新<br/>L = L_GRPO + λ·L_PSD，λ = 0.1"]
    D --> G
    G --> H["更新策略 π_k+1"]
    H -->|"产生新的未解决失败分布，进入下一轮（实验 K=2）"| A
    H --> I["部署只保留学生分支"]
```

图依据：论文 §3.1–3.5、Fig. 2。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.13395](https://arxiv.org/abs/2608.13395) |
| **文内评测** | nuScenes 构建 |
| **开源** | **待核实**（2026-09-26） |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。

## 实验与评测

- **文内口径：** nuScenes 构建
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [Temporal GRPO](./paper-temporal-grpo.md) | 同周「训练范式」线：阶段信用 vs 失败驱动 teacher（若适用） |
| [VLA 方法页](../methods/vla.md) | 本页为单篇索引；机制细节以原文为准 |

## 结论

**FIRE-VLA 适合作为本期「训练范式/智驾」路线的快速索引页。**

1. 核心贡献：组内轨迹都很差时 GRPO 难找正确方向；挑出相似失败，用冻结旧模型作 teacher 沿新模型输出继续纠正，配合常规范 RL，减少严重出错与反复失败。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第三篇）](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part3.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md)
- [arXiv:2608.13395](https://arxiv.org/abs/2608.13395)
- [arXiv PDF](https://arxiv.org/pdf/2608.13395)：已通读方法部分（实验与附录以略读为准）：§3.1–3.5（含 Fig. 2）

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.13395)
