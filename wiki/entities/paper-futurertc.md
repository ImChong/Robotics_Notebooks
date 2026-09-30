---
type: entity
tags:
  - paper
  - vla
  - action-chunking
  - asynchronous-deployment
  - plug-and-play
  - scu
  - uestc
status: complete
updated: 2026-09-30
arxiv: "2607.24008"
related:
  - ../methods/action-chunking.md
  - ../methods/vla.md
  - ./paper-real-time-chunking.md
  - ./paper-training-time-real-time-chunking.md
  - ./paper-remac.md
  - ./lerobot.md
  - ./paper-wam-realtime-async.md
sources:
  - ../../sources/papers/futurertc_arxiv_2607_24008.md
  - ../../sources/sites/futurertc-proj.md
summary: "FutureRTC（arXiv:2607.24008，川大/UESTC/Alberta）：冻结 VLA 前挂 anticipatory adapter，预测执行时刻视觉 latent 与本体状态，缓解 prediction–execution misalignment；LIBERO d=20 上 π₀.₅ 88.5% vs naive async 68.3%；截至入库日代码待发布。"
---

# FutureRTC（Anticipatory-Conditioned Chunking · arXiv:2607.24008）

**FutureRTC**（*Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking*，[arXiv:2607.24008](https://arxiv.org/abs/2607.24008)，[项目页](https://jianghaiscu.github.io/FutureRTC_proj/)）由 **四川大学**、**电子科技大学（UESTC）**、**阿尔伯塔大学** 等提出：在 **不修改冻结 VLA 权重** 的前提下，用轻量 adapter 预测 **执行时刻** 的 \((\hat z,\hat s)\)，再调用原策略生成 chunk，针对异步部署的 **prediction–execution misalignment**（比单纯修边界或只 forward 状态更根本）。

## 一句话定义

> **异步时观测必然 stale——FutureRTC 用 motion-aware 特征搬运 + 状态残差补全，把「执行那一刻该看到的 latent」补出来再喂冻结 VLA。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FutureRTC | Anticipatory-Conditioned Action Chunking | 本文 plug-and-play 框架 |
| SCM | State Correction Module | 对 roll-forward 状态的 MLP 残差 |
| OPM | Observation Prediction Module | 视觉 latent 空间 warp + synthesis |
| VLASH | — | 只 forward 本体状态的 training-time 对照 |
| RTC | Real-Time Chunking | 推理期 inpainting 对照 |

## 为什么重要

- **Oracle 实验定因：** 项目页报告若喂 **真执行时刻** \((o,s)\)，LIBERO 成功率随 delay 几乎平坦（~96.6% @ π₀.₅）——性能掉点主要来自 **条件输入陈旧**，而非异步本身。
- **同时动视觉与状态：** [VLASH](https://arxiv.org/abs/2512.01031) 类方法只补 proprio；机械臂运动导致 **视野内容变化**，FutureRTC 在 **VLA 视觉 latent** 上预测。
- **开销可控：** 约 +5.19M 参数、+3.04 ms（π₀.₅ 档），相对远程 VLA forward 仍小。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 四川大学；UESTC；University of Alberta |
| **Backbone** | LeRobot 微调 π₀.₅、SmolVLA-450M 等 **冻结** |
| **开源** | **待发布**（项目页：Under review · code will be released） |

## 核心原理

1. **State Correction Module：** \(\tilde s\) 由 stale 状态 + 已提交动作 roll-forward；SCM 预测残差 \(\tilde s_\Delta\) 对齐真 \(s_{t+K}\)。
2. **Observation Prediction Module：** 将 committed actions 编码为 motion prior → 2D flow + transport gate α warp stale 视觉特征；synthesis gate β 补接触 / 新露出区域；**绕过** vision encoder 直送 policy。
3. **Policy consistency loss：** 预测上下文下单步 flow 近似，使 chunk 接近「真执行时刻输入」下的输出。

## 评测（项目页摘要）

- **LIBERO** 四套件均值：π₀.₅ 在 \(d=20\) **88.5%** / 175.1 steps（naive async 68.3%）；优于 RTC、T-RTC、VLASH、REMAC 等同行数字（同一 frozen 权重协议，training-time 基线按论文复现）。
- **Kinetix 12 任务：** \(d=0\to4\) 成功率仅降 ~3.0%；步数在 \(d\ge2\) 最少。
- **双臂 AgileX + π₀.₅ @ 30 Hz：** 远程 ~170 ms（\(d\approx5\)）与 +150 ms 注入；三任务各 20 trial，延迟升高时 margin 扩大（如 Fold Towel @ 320 ms **80% vs 35%** naive）。

## 与其他工作对比

| 维度 | FutureRTC | 对照 |
|------|-------------|------|
| 修复位置 | **条件输入**：预测执行时刻的视觉 latent 与本体状态，再喂冻结 VLA | [Real-Time Chunking](./paper-real-time-chunking.md)：**动作空间** 推理期 inpainting，锁定已提交前缀、补全其余步 |
| 是否改 VLA 权重 | 不改；只训练约 +5.19M 参数 adapter（数值摘自项目页） | [Training-Time RTC](./paper-training-time-real-time-chunking.md)：训练时随机 delay 并对 action prefix 条件化，需重训策略 |
| 推理开销 | 约 +3.04 ms（π₀.₅ 档，数值摘自项目页） | [REMAC](./paper-remac.md)：训练期 masked chunking + prefix-preserving 采样，推理无额外延迟 |
| 关系 | 改上下文，与改 policy 的方法正交 | [REMAC](./paper-remac.md) / [Training-Time RTC](./paper-training-time-real-time-chunking.md)：改 policy 本身；LIBERO 对比表中作为同协议对照 |

## 结论

**异步 chunk 的第一性修复是「执行时刻条件输入」，不是只在 action 空间抹平；FutureRTC 用 adapter 逼近 oracle 条件，且不动 VLA 权重。**

- 适合已有 LeRobot checkpoint、不愿全量重训 VLA 的团队
- 代码未发布前仅可复现思想与对照数字，细节以 PDF / 项目页为准
- 与 [Training-Time RTC](./paper-training-time-real-time-chunking.md) / [REMAC](./paper-remac.md) **正交**：后者改 policy，FutureRTC 改 **上下文**
- 真机视频显示 chunk 边界 jerk 明显少于 RTC / VLASH（项目页 Hang Cups 270 steps / 26.5 s vs 31 s 级）

## 源码运行时序图

**不适用**（截至 2026-09-30）：项目页声明 code will be released；无官方可运行仓库。放出后预期链：**stale \((z,s)\) + committed actions → SCM/OPM → 冻结 `predict_action_chunk` → 10–30 Hz 执行**。

## 局限与风险

- Adapter 在 latent 空间外推，极端遮挡 / 非刚性场景可能失效（论文 Limitations）。
- 对比表混合 **inference-time** 与 **training-time** 方法，读榜时分开看是否改权重。
- 阿尔伯塔大学等 tag 未注册于 `institutions.json`。

## 关联页面

- [Real-Time Chunking](./paper-real-time-chunking.md)
- [REMAC](./paper-remac.md)
- [LeRobot](./lerobot.md)
- [Action Chunking](../methods/action-chunking.md)

## 参考来源

- [futurertc_arxiv_2607_24008](../../sources/papers/futurertc_arxiv_2607_24008.md)
- [futurertc-proj](../../sources/sites/futurertc-proj.md)

## 推荐继续阅读

- [arXiv:2607.24008](https://arxiv.org/abs/2607.24008)
- [FutureRTC 项目页](https://jianghaiscu.github.io/FutureRTC_proj/)
