---
type: method
tags: [locomotion, perception, reinforcement-learning, parkour, depth, legged]
status: complete
original: ../../sources/papers/pie_arxiv_2408_13740.md
updated: 2026-11-20
summary: "PIE（Parkour with Implicit-Explicit Learning）单阶段端到端 RL：深度图+本体经多头估计器输出显式高度图/速度/足端离地量与隐式环境表征，实现低成本四足挑战地形零样本跑酷。"
related:
  - ../entities/paper-cref.md
  - ../queries/robot-perception-stack-selection-loop.md
  - ./dreamwaq.md
  - ../entities/dreamwaq-plus.md
  - ../entities/extreme-parkour.md
  - ../concepts/privileged-training.md
  - ../concepts/terrain-adaptation.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ../overview/humanoid-rl-motion-control-methods.md
  - ../entities/paper-p3.md
  - ../entities/paper-mgdp-generalized-depth-perception.md
sources:
  - ../../sources/papers/pie_arxiv_2408_13740.md
  - ../../sources/papers/humanoid_motion_control_know_how.md
  - ../../sources/papers/p3_arxiv_2607_25541.md
  - ../../sources/papers/cref_arxiv_2603_29452.md
---

# PIE：感知一阶段鲁棒行走（隐式–显式学习）

**PIE**（*Parkour with Implicit-Explicit Learning Framework for Legged Robots*，ZJU，IEEE RA-L 2024，[arXiv:2408.13740](https://arxiv.org/abs/2408.13740)）是**单阶段**视觉跑酷框架：部署时仅依赖深度图与本体观测；估计器（CNN+MLP+共享 Transformer+GRU）输出**显式**物理量（高度图编码 z^m、基座速度、足端离地高度）与**隐式**VAE latent z（并重建下一时刻本体 o_{t+1}），与 PPO 并行训练，Actor 以 45 维本体 + 这些估计量为输入。RoboParty 飞书 Know-How 将其列为「**PIE 感知一阶段鲁棒行走**」教学模块。

## 一句话定义

不拆 teacher–student 两阶段，让深度+本体在训练期直接为最终动作服务，同时维护可解释的显式地形估计与压缩的隐式环境表征。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PIE | Parkour with Implicit-Explicit Learning | 本文框架名 |
| RL | Reinforcement Learning | 单阶段 PPO 训练 |
| VAE | Variational Autoencoder | 隐式环境表征常用组件 |
| Sim2Real | Simulation to Real | 论文报告零样本真机跑酷 |
| AMP | Adversarial Motion Prior | 可与运动先验结合（社区复现路线） |
| MPC | Model Predictive Control | 模型基线对照，非 PIE 组件 |
| GRU | Gated Recurrent Unit | 估计器中融合深度与本体、形成记忆 |
| KL | Kullback–Leibler divergence | 隐式 latent 的 VAE 正则项 |

## 为什么重要

- **相对盲走 DreamWaQ**：显式高度图提供**前瞻**，减少「先撞再改」的楼梯/缺口失败。
- **相对 Extreme Parkour 两阶段蒸馏**：单阶段避免 student 只模仿 teacher 行为、视觉表征约束不足。
- **工程教学价值**：飞书把 PIE 与 DreamWaQ、Attention 落足并列为感知 loco 三条线。

## 核心原理

```mermaid
flowchart LR
  PROP["本体历史 o_t (H1=10)"] --> MLP[MLP]
  DEPTH["深度图 (H2=2)"] --> CNN[CNN]
  MLP --> TR[共享 Transformer]
  CNN --> TR
  TR --> GRU[GRU 记忆]
  GRU --> ZM["z^m → 解码高度图 m̂"]
  GRU --> ZV["z (VAE) → 重建 o_{t+1}"]
  GRU --> VF["v̂ 基座速度 / ĥ_ft 足端离地"]
  ZM --> ACT["Actor (+45 维 o_t) → 12 关节偏置"]
  ZV --> ACT
  VF --> ACT
  CRIT["Critic (+特权 v_t, 高度图 m_t)"] -.训练期.-> PPO[PPO]
  ACT --> PPO
  GRU -.回归 + KL 损失.-> LOSS[Estimator Loss]
```

- **显式头：** 可监督回归，便于调试与 sim2real 诊断。
- **隐式头：** 捕捉摩擦、材质等难显式建模因素。
- **部署：** 仅深度 + 本体 + 估计器 + Actor；论文平台 DEEP Robotics Lite3（12.7kg），D435i 深度 10Hz、动作 50Hz、板载 RK3588。
- **训练：** Isaac Gym 4096 环境，RTX 4090 上 10,000 iter 不到 20 小时；奖励为类盲走的简单项（Table I）。

## 主要技术路线

| 路线 | 代表链接 | 说明 |
|------|----------|------|
| 单阶段估计 | [Terrain Adaptation](../concepts/terrain-adaptation.md) | 感知–本体融合 |
| 特权训练 | [Privileged Training](../concepts/privileged-training.md) | 非对称 critic |
| 对照 | [Extreme Parkour](../entities/extreme-parkour.md) | 两阶段视觉跑酷 |

## 工程实践

1. 从 LeggedGym / parkour 开源栈理解两阶段视觉跑酷基线。
2. 将 Estimator 回归损失与 PPO 策略损失**同阶段**优化（飞书强调与两阶段对比）。
3. 人形迁移：改观测维度、足端几何与奖励；勿直接照搬四足高度图分辨率。

## 实验结果（论文）

- 仿真 Table III（平均终止难度等级）：完整 PIE gap/stairs/step = 9.9/9.86/9.81；去掉 ĥ_ft 降至 7.41/7.36/3.09；去掉 m̂ 台阶/阶梯 1.67/4.25；去掉 o_{t+1} 重建在相机噪声大时差距显著。
- 真机：跨越 0.75m 台阶（约 3× 机身高）、1m 间隙（约 3× 机身长）、0.25m 楼梯；Table IV 相对前作至少提升 50%；室外 2km/爬升 153m 约 40 分钟完成。

## 局限与风险

- **深度质量敏感**：快速运动、强光、运动模糊会削弱估计（论文针对低成本相机设计）。
- **跑酷域与日常行走**：奖励与地形分布偏极限障碍；办公室平地需重训或微调。
- **原文验证平台为四足**：人形需重新验证质心高度与接触序列。
- **论文自述局限：** 仅深度、无 RGB 语义；只在静态环境训练；无 3D 理解，不能蹲下钻过障碍。
- **源码：** 论文未给出官方代码链接，本页未做源码核对，故无源码运行时序图（不适用）。

## 结论

PIE 的核心价值是以单阶段 PPO + 隐式/显式双层估计，把低成本四足的视觉跑酷做稳。

- 显式 ĥ_ft 与 m̂ 回归最关键（消融下降最大），不要省。
- 重建 o_{t+1} 的隐式头在相机噪声/错位时让策略更信任本体。
- 单阶段避免两阶段蒸馏信息损失，奖励可保持简单。
- 迁移到人形需重做观测、奖励与地形分布；论文仅验证四足。
- 论文无官方源码，复现需自行实现估计器结构。

## 关联页面

- [DreamWaQ](./dreamwaq.md)、[DreamWaQ++](../entities/dreamwaq-plus.md) — 盲走与多模态扩展
- [Extreme Parkour](../entities/extreme-parkour.md) — 两阶段视觉跑酷对照
- [AME 论文](../entities/paper-ame-attention-based-map-encoding.md) — Attention 地形编码姊妹线
- [P³](../entities/paper-p3.md) — 外感知进 VAE 后，PPO 应对边缘策略做 clip（论文引用 PIE 为 VAE-PPO 谱系）
- [CReF](../entities/paper-cref.md) — 同组后续人形单阶段 raw 深度；**去掉**显式高度图头，改本体查询注意 + 落脚奖励
- [MGDP](../entities/paper-mgdp-generalized-depth-perception.md) — HKU/BIT 对比学习深度+高程、两阶段解耦与 9 构型跨平台微调（*Adv. Sci.* 2026；已开源）
- [Know-How 技术地图](../overview/humanoid-motion-control-know-how-technology-map.md)

## 参考来源

- [pie_arxiv_2408_13740.md](../../sources/papers/pie_arxiv_2408_13740.md)
- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)
- [P³ 论文摘录（arXiv:2607.25541）](../../sources/papers/p3_arxiv_2607_25541.md)
- [CReF 论文摘录（arXiv:2603.29452）](../../sources/papers/cref_arxiv_2603_29452.md)

## 推荐继续阅读

- [arXiv:2408.13740](https://arxiv.org/abs/2408.13740)
- [Privileged Training](../concepts/privileged-training.md) — 非对称 AC 与估计器谱系
