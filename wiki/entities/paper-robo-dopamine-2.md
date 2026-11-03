---
type: entity
tags:
  - paper
  - vla
  - process-reward
  - rl
  - manipulation
status: complete
updated: 2026-09-27
arxiv: "2608.15680"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md
sources:
  - ../../sources/papers/robo-dopamine-2_arxiv_2608_15680.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md
summary: "Robo-Dopamine 2.0（arXiv:2608.15680）：历史条件 + OOD 感知的视觉过程奖励：对照参考过程判断继续/变化/失败/恢复，先学顺序再学进度并用异常轨迹校验，辅助 RL 改进 VLA。"
---

# Robo-Dopamine 2.0（arXiv:2608.15680）

**Robo-Dopamine 2.0**（*Robo-Dopamine 2.0: History-Conditioned and OOD-Aware Process Reward Modeling for Robotic Manipulation*，[arXiv:2608.15680](https://arxiv.org/abs/2608.15680)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第四篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md) **架构模块** 段。

## 一句话定义

**历史条件 + OOD 感知的视觉过程奖励：对照参考过程判断继续/变化/失败/恢复，先学顺序再学进度并用异常轨迹校验，辅助 RL 改进 VLA。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| PRM | Process Reward Model | 过程/进度奖励模型 |
| OOD | Out-of-Distribution | 分布外场景或轨迹 |

## 为什么重要

- 稀疏成功奖励与只看前后帧的 PRM 难区分正常变化与失败；2.0 显式引入历史与参考过程对齐。
- 策展机构：北京大学；EvoPhys AI；北京邮电大学；腾讯；中国人民大学；中山大学
- 开源结论：**待核实**（步骤 2.5，2026-09-27）。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["训练数据构造"]
        SRC["成功源轨迹<br/>RoboCasa / LIBERO / AgiBot World"] --> EDIT["task/phase-aware 编辑算子 E_z<br/>鲁棒扰动 R（遮挡/背景/干扰物/非目标颜色）<br/>任务无效 N（错物体/空抓/不释放/换目标）"]
        EDIT --> SIGN["有符号进度 Φ<br/>Φ(P)=Φ(R)=g_i，Φ(N)=-g_i"]
        SIGN --> HOP["Signed-Hop 目标<br/>y = 100·clip(H, -1, 1)"]
        HOP --> PAIR["Pair compiler 编译 P/R/N 间的<br/>有效、失败、恢复、不变配对"]
    end
    subgraph CUR["Signed-Hop 两阶段课程（共 400K 样本）"]
        S1["Stage 1：大 Hop 池 D_large<br/>学全局进度/失败/恢复几何"] --> S2["Stage 2：以细 Hop/零 Hop 为主<br/>回放 ρ=0.25 的大 Hop 样本"]
    end
    PAIR --> S1
    S2 --> GRM["GRM（General Reward Model）<br/>输入：指令 c、起止锚点、有序时间 panel、成对状态 (X_b, X_a)<br/>输出：有符号相对进度 ŷ_ba（MSE 回归）"]
    subgraph ONLINE["闭环推理"]
        Q["三类查询：X_0→X_t（前向）、X_G→X_t（后向）、X_t-1→X_t（增量）<br/>时间 panel 取自已观测 rollout 历史"] --> GRM
        GRM --> FUSE["逆 Hop 还原三个势函数估计<br/>取平均得 Φ̂_t"]
        FUSE --> RW["稠密奖励 r_t = r_task + γ(Φ̂_t+1 − Φ̂_t)"]
        RW --> RL["下游强化学习"]
        RL --> ROLL["rollout 观测"]
        ROLL --> Q
    end
```

图依据：论文 Method 各小节（History-Conditioned Pairwise Reward / Structured OOD Data / Signed Progress Space / Signed-Hop Curriculum / Closed-Loop Reward Framework）/ Fig.1、式(1)–(18)

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.15680](https://arxiv.org/abs/2608.15680) |
| **开源** | **待核实** |
| **文内评测** | RoboTwin；双 Franka 实机 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** RoboTwin；双 Franka 实机
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第四篇技术地图](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md) | 同批 14 篇横向索引；本文属 **架构模块** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**Robo-Dopamine 2.0 适合作为本期「架构模块」路线的快速索引页。**

1. 核心贡献：历史条件 + OOD 感知的视觉过程奖励：对照参考过程判断继续/变化/失败/恢复，先学顺序再学进度并用异常轨迹校验，辅助 RL 改进 VLA。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第四篇技术地图](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第四篇）](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part4.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md)
- [arXiv:2608.15680](https://arxiv.org/abs/2608.15680)
- [arXiv PDF](https://arxiv.org/pdf/2608.15680)：已通读方法部分（实验与附录以略读为准）：Method 全部小节（含式 1–18、Fig.1）；Related Work、实验与附录以略读为准

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.15680)
