---
type: task
tags: [humanoid, locomotion, whole-body-control]
status: complete
updated: 2026-07-21
related:
  - ./locomotion.md
  - ./stair-obstacle-perceptive-locomotion.md
  - ../concepts/terrain-adaptation.md
  - ../entities/paper-gaitspan-humanoid-locomotion-walking-running.md
  - ../entities/paper-roller-skating-amp-humanoid-passive-wheels.md
  - ../entities/paper-now-you-see-that-humanoid-vision-locomotion.md
  - ../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md
summary: "人形机器人在复杂地形下的平衡与移动任务，强调高维动力学处理、环境感知以及全身肢体协调。"
---

# Humanoid Locomotion (人形机器人移动)

**Humanoid Locomotion**：使双足类人机器人能够在复杂、非结构化的地形中，保持平衡的同时实现高效、鲁棒的位移，并具备全身协调（Whole-body Coordination）能力。

## 一句话定义

让两条腿（甚至加上手和膝盖）在各种烂路上走稳、走远、走得像人。

## 核心挑战

1. **高维非线性动力学**：人形机器人具有数十个自由度，其动力学模型高度复杂且存在欠驱动（Under-actuated）阶段。
2. **接触力学建模**：涉及足端、手部或膝盖与地形的断续接触，传统的基于模型的控制（如 MPC）在处理多点接触时计算量巨大。
3. **环境感知与反应**：需要将高程图（Elevation Maps）或点云信息实时转化为运动规划，以应对楼梯、斜坡和障碍物。

## 主流技术路线

### 1. 基于模型的控制 (Model-based Control)
- **核心**：利用简化模型（如 单质点模型 CoM, 线性倒立摆 LIP）进行轨迹规划，配合全身控制（WBC）进行任务分解。
- **代表作**：MIT Cheetah 系列的变体，IHMC 的双足控制。

### 2. 层级强化学习 (Hierarchical RL)
- **核心**：分层架构，高层负责技能规划（Skill Planning），底层负责电机指令跟踪。
- **趋势**：通过奖励函数让机器人自主探索步态，解决非线性接触问题。
- **技能生长（skill growth）：** [GaitSpan](../entities/paper-gaitspan-humanoid-locomotion-walking-running.md)（arXiv:2607.12114）把 **冻结行走策略** 当种子，用 GaitWave + H-SLIP + 残差在 **无人体演示** 下让走–慢跑–跑 **连续涌现**，覆盖 Booster T1/K1 与 Unitree G1 真机户外地形。

### 3. 生成式运动模型 (Generative Motion Models)
- **核心**：利用扩散模型（Diffusion Models）从人类数据中学习自然的运动先验。
- **进展**：ETH Zurich 的工作证明了扩散模型可以作为高效的实时全身运动生成器。

## 全身移动 (Whole-body Locomotion)

现代研究强调利用全身各个部位进行移动：
- **接触辅助**：在攀爬高箱时使用手臂辅助。
- **重心调节**：通过挥动手臂来补偿角动量。
- **环境自适应**：利用膝盖或身体侧面在狭窄空间支撑。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| CoM | Center of Mass | 质心，平衡与 locomotion 规划的核心状态量 |
| LIP | Linear Inverted Pendulum | 线性倒立摆，质心动力学的常用简化模型 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| MoCap | Motion Capture | 动作捕捉，参考动作与演示数据的主要来源 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 参考来源
- [Chasing Autonomy: Dynamic Retargeting and Control Guided RL for Performant and Controllable Humanoid Running](../../sources/papers/chasing_autonomy.md)
- [SPRINT: Efficient Spectral Priors for Humanoid Athletic Sprints](../../sources/papers/sprint_arxiv_2605_28549.md) — 5 条参考 + 频谱先验 + 残差 RL，G1 真机冲刺 6 m/s。
- [SSR: Scaling Surefooted and Symmetric Humanoid Traversal to the Open World](../../sources/papers/ssr_arxiv_2605_30770.md) — 第一视角深度单阶段 PPO + 想象落脚点，AgiBot X2 户外 1.3 km 长程。
- [sources/papers/eth-g1-diffusion.md](../../sources/papers/eth-g1-diffusion.md) — 基于扩散模型与 RL 的全身移动框架。
- [sources/papers/humanoid_hardware.md](../../sources/papers/humanoid_hardware.md) — 人形机器人硬件平台综述。
- [QuietWalk（arXiv:2604.23702）](../../sources/papers/quietwalk_arxiv_2604_23702.md) — PINN 估计竖直 GRF + RL 冲击惩罚，G1 跨鞋型低噪行走。
- [GaitSpan（arXiv:2607.12114）](../../sources/papers/gaitspan_arxiv_2607_12114.md) — 行走种子 + GaitWave/H-SLIP 技能生长，单策略连续走–慢跑–跑，五 embodiment 与户外零样本。
- [被动轮轮滑 AMP（arXiv:2607.10815）](../../sources/papers/roller_skating_amp_arxiv_2607_10815.md) — Booster T1 被动轮滑，切片圆柱轮仿真 + 双 gait AMP-PPO，Pump/Push Glide 真机验证。
- [RAVEN（arXiv:2607.15701）](../../sources/papers/raven_rl_adaptive_visibility_graph_arxiv_2607_15701.md) — RL 自适应可见图膨胀 + DAVG-cfMPC + Booster Gym，延迟下人形导航。

## 关联页面
- [Learning Whole-Body Humanoid Locomotion（ETH G1）](../entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) — 扩散运动生成 + RL 全身跟踪，真机箱攀/跨栏/楼梯与混合地形
- [SPRINT 人形竞技冲刺频谱先验](../entities/paper-sprint-humanoid-athletic-sprints.md) — 极少 MoCap + 频域先验外推至高动态冲刺
- [SSR 开放世界人形穿越](../entities/paper-ssr-humanoid-open-world-traversal.md) — 想象落脚点 + 潜空间对称 + 分地形 AMP，楼梯/沟壑/高台与户外长程
- [Now You See That 端到端视觉人形 locomotion](../entities/paper-now-you-see-that-humanoid-vision-locomotion.md) — 8 步立体深度增广 + 多 critic/discriminator 特权 RL + vision-aware DAgger 蒸馏，双向长楼梯与跑酷零样本
- [QuietWalk 物理感知低噪行走](../entities/paper-quietwalk-humanoid-locomotion.md) — 逆动力学 PINN 估计 GRF 作冲击惩罚；G1 真机 1.2 m/s 降噪约 7 dB，跨赤脚/运动鞋/高跟鞋与多地面材质
- [GaitSpan 从行走到跑步的技能生长](../entities/paper-gaitspan-humanoid-locomotion-walking-running.md) — 冻结行走种子 + GaitWave 节律组合 + H-SLIP 动态步幅；Booster T1/K1、G1 真机户外走–慢跑–跑连续变速
- [被动轮人形轮滑 AMP（Tsinghua）](../entities/paper-roller-skating-amp-humanoid-passive-wheels.md) — 被动轮滑 + 9 片圆柱碰撞模型；人体 MoCap→GMR→独立 AMP 学 Pump Glide / Push Glide
- [RAVEN：RL 自适应可见图 + cf-MPC](../entities/paper-raven-rl-adaptive-visibility-graph-mpc.md) — 导航层 RL 改障碍膨胀，行走层 Booster Gym；延迟与噪声下鲁棒导航
- [Chasing Autonomy Pipeline](../methods/chasing-autonomy-pipeline.md) — 结合重定向与控制引导的 RL 实现高性能奔跑
- [楼梯与障碍感知移动](./stair-obstacle-perceptive-locomotion.md) — 带/不带感知的上下楼梯与越障挂接点
- [Locomotion](./locomotion.md)
- [ZEST](../methods/zest.md) — Boston Dynamics 跨形态高动态模仿与零样本部署
- [MTRG / GfR](../methods/mtrg-reference-goal-driven-rl.md) — RSS 2026；G1 箱式跑酷：参考塑形 + goal 泛化（超越 ZEST tracking 的 OOD 鲁棒性）
- [HIL](../methods/hil-hybrid-imitation-learning.md) — 物理角色跑酷：tracking + AMP 混合模仿（仿真）
- [HIL vs MTRG vs ZEST 跑酷路线对比](../comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md) — 跑酷模仿三条路线选型
- [Diffusion-based Motion Generation](../methods/diffusion-motion-generation.md)
- [PPO](../methods/policy-optimization.md)
- [Whole-Body Coordination](../concepts/whole-body-coordination.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)

## 推荐继续阅读

- [机器人论文阅读笔记：Now You See That](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Now_You_See_That_Learning_End-to-End_Humanoid_Locomotion_from_Raw_Pixels/Now_You_See_That_Learning_End-to-End_Humanoid_Locomotion_from_Raw_Pixels.html)
- [Now You See That 项目页](https://hellod035.github.io/Now_You_See_That/) — RSS 2026；立体深度增广与实机跑酷/楼梯 demo
- [机器人论文阅读笔记：HoRD](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/HoRD__Robust_Humanoid_Control_via_History-Conditioned_RL_and_Online_Distillation/HoRD__Robust_Humanoid_Control_via_History-Conditioned_RL_and_Online_Distillation.html)
