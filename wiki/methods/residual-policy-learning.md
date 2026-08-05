---
type: method
tags: [residual-learning, reinforcement-learning, control, locomotion, manipulation, motion-tracking, sim2real, shared-autonomy]
status: complete
updated: 2026-08-05
related:
  - ./reinforcement-learning.md
  - ./imitation-learning.md
  - ./deepmimic.md
  - ../concepts/sim2real.md
  - ../concepts/whole-body-tracking-pipeline.md
  - ../tasks/locomotion.md
  - ../entities/paper-loco-manip-161-157-refine-dp.md
  - ../entities/paper-notebook-robotdancing-residual-action-rl-enables-robust-l.md
sources:
  - ../../sources/personal/residual-policy-reading-list.md
  - ../../sources/papers/refine_dp_arxiv_2603_13707.md
  - ../../sources/papers/robotdancing_arxiv_2509_20717.md
summary: "Residual Policy Learning（残差策略学习）：最终动作 = 基础动作 + 学习残差，a=a_base+Δa。基础部分可以是传统控制器、MPC、参考轨迹、技能解码器、运动生成器甚至人的输入；RL 只学补偿量，从而收窄探索空间、保住 base 先验、提升样本效率。本页给出统一形式、十篇代表论文谱系与选型建议。"
---

# Residual Policy Learning（残差策略学习）

**Residual Policy Learning** 是一类「base + 残差」的机器人学习方法：最终动作由**已有的基础行为产生器**（传统控制器、MPC、参考轨迹、技能解码器、运动生成器、 Motion Tracking 策略，甚至人的操作命令）与**可学习的残差策略**相加而成，RL 只负责学习补偿量 $\Delta a$，处理 base 难以覆盖的接触、摩擦、模型误差与动力学失配。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RPL | Residual Policy Learning | Silver et al. 2018 正式命名的残差策略学习框架 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略，本页中指只训练残差部分 |
| MPC | Model Predictive Control | 常见的 base 控制器来源之一（已知或学习模型） |
| TD3 | Twin Delayed DDPG | Johannink Residual RL 与 Multi-Modal ARRL 使用的 off-policy 算法 |
| DDPG | Deep Deterministic Policy Gradient | Silver RPL 搭配 HER 的连续控制算法 |
| HER | Hindsight Experience Replay | 稀疏奖励目标条件任务的经验回放增强 |
| RFC | Residual Force Control | Yuan & Kitani：残差以根部外力形式注入动作空间 |
| ARRL | Automated Residual Reinforcement Learning | Yu & Rosendo：base 控制器参数由黑箱优化器同步自动训练 |
| RSA | Residual Shared Autonomy | Schaff & Walter：base policy 是人的共享自治残差框架 |
| CMG | Conditional Motion Generator | RuN 的运动先验生成器，残差叠加在其输出之上 |
| GMT | General Motion Tracking | ResMimic 的通用运动跟踪先验策略 |

## 为什么重要

- **纯 RL 的三座大山**：样本效率低、初期探索危险、稀疏奖励长视野任务难收敛；一个好但不完美的 base 把探索空间收窄到「修正量」量级，三座大山同时缓解。
- **传统控制的三座大山**：接触/摩擦难建模、参数需手调、行为无法从数据自适应；残差恰好补上这三块，且**不丢弃**控制器已有的稳定性结构。
- **统一的工程语言**：从 2018 年经典两篇到 2025 年 G1 人形工作（RuN/ResMimic），「$a=a_{\text{base}}+\Delta a$」贯穿机器人学习十年谱系，是读懂现代分层/残差架构（ASAP、RobotDancing、OmniTacTune 等）的钥匙。
- **真机可行性证据链完整**：3 小时真机训练（Johannink）、50 cm 连续跳跃（Yang）、2.5 m/s 走跑切换（RuN）、92.5% sim-to-sim 成功率（ResMimic）均建立在残差分解之上。

## 核心原理

### 统一形式

$$a_t = a_t^{\text{base}} + \Delta a_t,\qquad \Delta a_t \sim \pi_\theta(\cdot \mid s_t, \text{可选上下文})$$

- **Residual MDP 视角**（Silver et al.）：固定 base 策略 $\pi$ 与环境 MDP 诱导出残差 MDP $M^{(\pi)}$，其转移 $T^{(\pi)}(s,a,s')=T(s,\pi(s)+a,s')$；残差策略就是该 MDP 中的普通策略，任何连续动作 RL 算法均可训练。
- **奖励分解视角**（Johannink et al.）：奖励写成 $r=f(s_m)+g(s_o)$，$f$（机器人自身几何目标）由传统控制器高效优化，$g$（物体/接触相关目标）由 RL 残差学习。
- **探索偏置视角**：残差把初始状态分布与动作分布偏置到 base 的高回报区域附近，等效于「站在 base 肩膀上探索」。

### 工程三件套（谱系中反复出现）

1. **残差零初始化**：末层置零或小增益初始化，使训练初 $\Delta a\approx 0$，保证初始性能不差于 base（Silver、ResMimic 均用）。
2. **Value/critic burn-in**：先只训价值函数（策略固定为 base），避免初期差 critic 带坏好 base（Silver 的 $\beta$ 阈值、RSA 的 100K 步 warm-up）。
3. **残差幅值正则**：惩罚 $\|\Delta a\|$ 让策略「尽量不动 base」，在 RSA 中上升为显式约束目标（最小干预保持人的控制权）。

### 残差可以作用在不同空间

| 残差空间 | 代表工作 | 备注 |
|----------|----------|------|
| 控制量/关节动作 | Johannink、Silver、Jumping、RuN、ResMimic | 最常见，真机可直接部署 |
| 外部力/力矩 | RFC | **仿真特权**，真机无此外力 |
| 技能解码后的原子动作 | ReSkill | 分层：高层选技能、低层残差细修 |
| 人的输入通道 | RSA | base 是人，残差是辅助 |

## 主要技术路线：十篇代表论文谱系

| 论文 | 年份/出处 | Base 部分 | 残差输出 | 真机 | 开源 |
|------|-----------|-----------|----------|------|------|
| [Residual RL for Robot Control](../entities/paper-residual-rl-robot-control.md)（Johannink） | ICRA 2019 | 阻抗/位置反馈控制器 | 控制量修正（TD3） | Sawyer 装配 | 未开源 |
| [Residual Policy Learning](../entities/paper-residual-policy-learning.md)（Silver） | 2018 | 人工控制器 / MPC | Action 修正（DDPG+HER） | 仿真 | 已开源 |
| [RFC](../entities/paper-rfc-residual-force-control.md)（Yuan & Kitani） | NeurIPS 2020 | DeepMimic 系模仿策略 | 根部残差外力（PPO） | 仿真角色 | 已开源（非商用） |
| [Continuous Versatile Jumping](../entities/paper-versatile-jumping-action-residuals.md)（Yang） | L4DC 2022 | 加速度控制器 + WBC | 机身位姿/加速度修正（ARS） | Go1 跳跃 | 未开源 |
| [Multi-Modal ARRL](../entities/paper-multimodal-legged-arrl.md)（Yu & Rosendo） | RA-L/IROS 2022 | PD 控制器 + 开环步态（同步自动调参） | 关节角增量（TD3/SAC） | Mini Cheetah 双足 | 已开源 |
| [ReSkill](../entities/paper-reskill-residual-skill-policies.md)（Rana） | CoRL 2022 | VAE 技能解码器 + flows 技能先验 | 原子动作修正（on-policy） | 仿真 | 已开源（MIT） |
| [Residual Shared Autonomy](../entities/paper-residual-policy-shared-autonomy.md)（Schaff & Walter） | ICRA 2020 | **人的操作命令** | 最小干预修正（约束 PPO） | 仿真+人测 | 已开源 |
| [RuN](../entities/paper-notebook-run-residual-policy-for-natural-humanoid-locomot.md)（Li et al.） | 2025 | Conditional Motion Generator | 关节目标修正（PPO） | G1 走跑 2.5 m/s | 未开源 |
| [ResMimic](../entities/paper-resmimic.md)（Zhao et al.） | 2025 | GMT 通用跟踪策略 | 全身动作修正（PPO） | G1 搬运 4.5–5.5 kg | 已开源 |
| [RobotDancing](../entities/paper-notebook-robotdancing-residual-action-rl-enables-robust-l.md)（Sun et al.） | 2025/2026（RA-L） | Retarget 参考轨迹（选择性 DoF） | 髋/膝 pitch 残差目标（PPO） | G1 长时程舞蹈 21/24 | 未开源 |

**推荐阅读顺序**：1 → 2 建立基础思想；3 理解动作模仿中的动力学失配补偿；4 理解真实腿足机器人控制器打底；8 → 9 对应现代 G1 人形形态；10 看「长参考 + 选择性残差」在舞蹈追踪上的工程配方。

### 与其他残差类工作的边界

本谱系聚焦「base 行为 + 加性修正」主线。[RobotDancing](../entities/paper-notebook-robotdancing-residual-action-rl-enables-robust-l.md) 已纳入上表：base 是 **retarget 参考**（非学到的控制器/生成器），残差默认只开承重关键 DoF，并强调长尾采样——与 RuN/ResMimic「学到的先验 + 残差」同形、不同 base 来源。其余变体宜交叉阅读而非混为一谈：[ASAP](../entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md)（delta 动作模型学 sim–real 动力学差）、[OmniTacTune](../entities/paper-omnitactune-tactile-residual-adaptation.md)（触觉残差自适应）、[FARM](../entities/paper-notebook-farm-frame-accelerated-augmentation-and-residual.md)（帧加速增广与残差 MoE）、[Residual Off-Policy RL for BC Finetuning](../entities/paper-notebook-residual-off-policy-rl-for-finetuning-behavior-c.md)（BC 微调残差）。

## 工程实践

1. **什么时候用残差**：手头有 60–90 分的 base（控制器/参考/预训练策略），且失败模式集中在接触、扰动、模型误差等「难建模尾部」；base 完全不可用或任务与 base 无关时，残差无意义。
2. **base 选型决策**：有模型 → MPC/WBC 打底（Jumping）；有参考轨迹/数据 → Motion Generator 或 GMT 打底（RuN、ResMimic）；有现成技能库 → 技能解码器打底（ReSkill）；有人在场 → 人打底（RSA）。
3. **初始化与训练顺序**：残差末层零初始化 → value burn-in → 联合训练；base **冻结**（微调 base 会破坏先验，ResMimic Table I 显示微调甚至劣于残差）。
4. **残差输入可以宽于 base**：残差网络可以看到 base 看不到的信息（物体状态、人的原始指令、特权参考），这是残差 ≠ 简单增益调度的关键。
5. **部署检查**：确认残差空间在真机物理可实现（RFC 根部外力不可直接上真机）；确认 base 与残差的**控制频率**匹配（Jumping 500 Hz 管线 vs ARS 策略频率）。

## 局限与风险

- **base 天花板**：残差继承 base 的结构性缺陷（如 Jumping 只支持 pronking 四足同起同落步态）；base 与任务分布严重失配时残差也救不回来（ReSkill 局限节明示）。
- **双组件非平稳性**：base 与残差若同时训练（ARRL、ReSkill），目标非平稳，需要 on-policy 算法或专门优化器组合。
- **「残差能修一切」误区**：残差学的是补偿分布，不是新技能本身；新技能仍需要新的 base 或新的 Stage II 训练（ResMimic 每任务残差）。
- **仿真特权混淆**：力空间残差（RFC）适合仿真训练与内容生成，不能直接当作真机控制方案引用。

## 与其他方法对比

| 路线 | 先验利用方式 | base 可微？ | 探索空间 | 典型风险 |
|------|--------------|-------------|----------|----------|
| 纯 RL from scratch | 无 | — | 全动作空间 | 样本贵、初期危险、稀疏奖励失败 |
| 微调（fine-tuning） | 参数初始化 | 需要 | 全动作空间 | 灾难性遗忘 base 先验 |
| **Residual Policy** | **行为叠加** | **不需要** | **修正量级** | base 天花板、双组件非平稳 |
| 分层/技能 RL | 时间抽象 | 不需要 | 技能空间 | 技能空间不含解则失败（ReSkill 的动机） |
| [REFINE-DP](../entities/paper-loco-manip-161-157-refine-dp.md) 式联合 RLFT | 直接更新 DP 参数 + 同步低层跟踪器 | DP 可微（DPPO） | 规划命令空间 + 关节跟踪 | 规划–控制分布需一起拉齐；非加性残差 |

## 关联页面

- [Reinforcement Learning](./reinforcement-learning.md)
- [Imitation Learning](./imitation-learning.md)
- [DeepMimic](./deepmimic.md)
- [Sim2Real](../concepts/sim2real.md)
- [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md)
- [Safe Real-World RL Fine-Tuning](../concepts/safe-real-world-rl-fine-tuning.md)
- [MPC vs RL](../comparisons/mpc-vs-rl.md)
- [REFINE-DP（论文实体）](../entities/paper-loco-manip-161-157-refine-dp.md) — 直接微调 DP + 联合低层，对照冻结 DP + Residual RL

## 参考来源

- [Residual Policy / Residual RL 论文精读清单摘录](../../sources/personal/residual-policy-reading-list.md)
- Silver et al., *Residual Policy Learning*, arXiv:1812.06298 — [sources/repos/residual-policy-learning](../../sources/repos/residual-policy-learning.md)
- Johannink et al., *Residual Reinforcement Learning for Robot Control*, ICRA 2019 — [sources/sites/residualrl-github-io](../../sources/sites/residualrl-github-io.md)

## 推荐继续阅读

- RPL 项目页与视频：<https://k-r-allen.github.io/residual-policy-learning/>
- Residual RL 项目页（真机装配视频）：<https://residualrl.github.io/>
- RFC 项目页（芭蕾/空翻演示）：<https://www.ye-yuan.com/rfc>
