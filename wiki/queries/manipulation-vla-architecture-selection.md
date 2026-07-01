---
title: 操作 VLA 与视频-动作架构选型指南
type: query
status: complete
created: 2026-05-21
updated: 2026-07-01
summary: 在灵巧操作任务中，如何在 VLA、Video-Action Model、解耦动力学 VLA、世界模型与开源策略家族之间选型与组合。
sources:
  - ../../sources/blogs/wechat_shenlan_vla_github_repro_survey_2025.md
  - ../../sources/papers/rl_foundation_models.md
  - ../../sources/papers/defi_arxiv_2604_16391.md
  - ../../sources/papers/mimic_video_arxiv_2512_15692.md
---

> **Query 产物**：本页由以下问题触发：「做灵巧操作，VLA / 视频模型 / 世界模型 / 开源策略该怎么选？」
> 综合来源：[VLA](../methods/vla.md)、[mimic-video](../methods/mimic-video.md)、[DeFI](../methods/defi-decoupled-dynamics-vla.md)、[Manipulation](../tasks/manipulation.md)

# 操作 VLA 与视频-动作架构选型指南

## TL;DR 决策树

| 你的约束 | 推荐路线 | 方法页 |
|----------|----------|--------|
| 有大规模带动作标签机器人数据 | 端到端 VLA / 开源策略族 | [VLA](../methods/vla.md)、[π₀](../methods/π0-policy.md)、[π0.7](../methods/pi07-policy.md)、[STAR-VLA](../methods/star-vla.md)、[Pelican](../methods/pelican-unified-1.md) |
| 有人视频、缺动作标签 | 解耦前向/逆动力学 | [DeFI](../methods/defi-decoupled-dynamics-vla.md) |
| 强调语义-动力学一体潜计划 | Video-Action Model | [mimic-video](../methods/mimic-video.md) |
| 需要显式交互物理想象 | 灵巧世界模型 | [DWM](../methods/dwm.md) |
| 语言+接触+全身协调 | 接触感知 transformer 路线 | [CLAW](../methods/claw.md) |
| 双手灵巧 + 高频触觉反应 | 变频率 MoT + 触觉 mid-training | [T-Rex](../entities/paper-trex-tactile-reactive-dexterous-manipulation.md) |

---

## 架构族对比（简表）

| 族 | 代表方法 | 核心假设 | 主要代价 |
|----|----------|----------|----------|
| **静态 VLM + 动作头** | VLA, π₀, π0.7, STAR-VLA, Pelican | 视觉-语言先验可迁移到控制 | 推理延迟、chunk 异步 |
| **视频潜计划 + 轻量解码** | mimic-video | 视频扩散承载语义与动力学 | 两阶段训练与部署 |
| **GFDM+GIDM 解耦** | DeFI | 人视频无动作标签仍可预训练 | 多阶段预训练与适配器 |
| **像素世界模型** | DWM | 在模型内 planning 交互 | 仿真-真机 gap |
| **语言接触全身** | CLAW | 语言指令与接触模态联合 | 数据与 sim 对齐 |
| **触觉反应灵巧双手** | T-Rex | 慢 VLA 规划 + 快触觉残差；人预训练 + 触觉 play mid-training | 需触觉硬件与 mid-training 数据 |
| **通才 VLM planner + actor VLA** | Vesta + Gr00t-N1.6 | 单 checkpoint 覆盖 VLN/空间推理/子任务规划；actor 执行文本子任务 | 两级延迟、actor 错误仍主导失败 |

---

## 分场景推荐

### 桌面已知物体、已有机器人 demo

优先 [VLA](../methods/vla.md) 或成熟开源权重：[π₀ Policy](../methods/π0-policy.md)、[π0.7 Policy](../methods/pi07-policy.md)、[STAR-VLA](../methods/star-vla.md)、[Pelican Unified-1](../methods/pelican-unified-1.md)。部署细节见 [VLA 部署指南](./vla-deployment-guide.md)。

### 长程、多步、语言条件

[mimic-video](../methods/mimic-video.md) 的 video-action 路线适合需要**潜空间长程一致性**的任务；[DeFI](../methods/defi-decoupled-dynamics-vla.md) 强调 CALVIN / SimplerEnv 类长程指标。

**Planner + actor 分层（非端到端 VLA）：** [Vesta](../entities/paper-vesta-generalist-embodied-reasoning.md) 用 **image+text memory** 与 **CoT 子任务** 作 System-2，配对 **Gr00t-N1.6** 等 actor；真机记忆型任务 **+38.3%**（arXiv:2606.20905）。与 [SayCan](../methods/saycan.md) 同族但 **统一多认知轴 SFT**。

### 人视频预训练 → 机器人微调

[DeFI](../methods/defi-decoupled-dynamics-vla.md) 明确面向无动作标签人视频；与 [Foundation Policy 人形指南](./foundation-policy-for-humanoids.md) 中的 embodiment gap 讨论互补。

### 需要「在脑子里试交互」

[DWM](../methods/dwm.md) 适合接触丰富、希望用 world model 做 roll-out 的路线，常与下游 MPC/IL 组合而非单独端到端。

### 全身+接触+语言

[CLAW](../methods/claw.md) 面向语言条件下的接触与全身协调，与纯桌面 VLA 的假设不同，硬件与观测栈需单独评估。

---

## 常见误区

1. **VLA 名字相同、栈不同**：π 系列、STAR、Pelican 在动作空间、tokenizer、微调数据上不可直接互换权重。
2. **视频模型 = 低延迟**：视频骨干推理往往更重，需 action chunk / 异步执行（见 [VLA 与低级控制器融合](./vla-with-low-level-controller.md)）。
3. **世界模型可跳过 sim 验证**：DWM 仍依赖接触动力学建模质量。

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| DR | Domain Randomization | 训练时随机化仿真参数以提升跨域鲁棒迁移 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| IL | Imitation Learning | 从专家演示学习策略，奖励难定义时的主路线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 参考来源

- [深蓝具身智能：VLA GitHub 复现推荐](../../sources/blogs/wechat_shenlan_vla_github_repro_survey_2025.md) — 开源仓库入口与复现目标分组
- [RL 基础模型综述（RT / π / Octo）](../../sources/papers/rl_foundation_models.md)
- [DeFI 论文入库](../../sources/papers/defi_arxiv_2604_16391.md)
- [mimic-video 论文摘录](../../sources/papers/mimic_video_arxiv_2512_15692.md)

## 关联页面

- [VLA 开源复现景观（2025）](../overview/vla-open-source-repro-landscape-2025.md)
- [VLA](../methods/vla.md)、[mimic-video](../methods/mimic-video.md)、[DeFI](../methods/defi-decoupled-dynamics-vla.md)、[DWM](../methods/dwm.md)、[CLAW](../methods/claw.md)
- [π₀ Policy](../methods/π0-policy.md)、[π0.7 Policy](../methods/pi07-policy.md)、[STAR-VLA](../methods/star-vla.md)、[Pelican Unified-1](../methods/pelican-unified-1.md)
- [VLA 部署指南](./vla-deployment-guide.md)、[IL for Manipulation](./il-for-manipulation.md)、[接触丰富操作指南](./contact-rich-manipulation-guide.md)

## 一句话记忆

> **有机器人动作数据走 VLA/π/STAR/Pelican；有人视频走 DeFI；要长程语义-动力学一体走 mimic-video；要物理想象走 DWM；要语言+接触全身走 CLAW；要双手灵巧高频触觉反应走 T-Rex。**
