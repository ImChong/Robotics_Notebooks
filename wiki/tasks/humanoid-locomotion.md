---
type: task
tags: [humanoid, locomotion, whole-body-control]
status: complete
updated: 2026-09-05
related:
  - ./locomotion.md
  - ./stair-obstacle-perceptive-locomotion.md
  - ../entities/paper-cref.md
  - ../concepts/humanoid-policy-observation-inputs.md
  - ../concepts/humanoid-policy-reward-functions.md
  - ../concepts/terrain-adaptation.md
  - ../entities/paper-humoslope-physics-guided-slope-locomotion.md
  - ../entities/paper-gaitspan-humanoid-locomotion-walking-running.md
  - ../entities/paper-roller-skating-amp-humanoid-passive-wheels.md
  - ../entities/paper-now-you-see-that-humanoid-vision-locomotion.md
  - ../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md
  - ../entities/paper-light-loco-parkour.md
  - ../entities/paper-parkourformer.md
  - ../entities/paper-adp.md
  - ../entities/paper-learning-quiet-walking-aibo.md
  - ../entities/paper-quietwalk-humanoid-locomotion.md
  - ../entities/paper-stay-seated.md
  - ../entities/paper-g1-confined-space-wbp.md
  - ../entities/paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md
  - ../entities/paper-p3.md
  - ../entities/paper-wm-loco.md
  - ../entities/paper-safe-stop-humanoid.md
  - ../entities/paper-bridge-humanoid.md
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
- **陡坡物理引导：** [HumoSlope](../entities/paper-humoslope-physics-guided-slope-locomotion.md)（arXiv:2607.07830）用 **局部支撑平面 ZMP 正则 + BSGA 生物力学软先验** 抑制低 CoM 蹲姿，G1 **纯本体感知** 户外草地坡至 **32.1°**。

### 3. 生成式运动模型 (Generative Motion Models)
- **核心**：利用扩散模型（Diffusion Models）从人类数据中学习自然的运动先验。
- **进展**：ETH Zurich 的工作证明了扩散模型可以作为高效的实时全身运动生成器。

## 全身移动 (Whole-body Locomotion)

现代研究强调利用全身各个部位进行移动：
- **接触辅助**：在攀爬高箱时使用手臂辅助。
- **重心调节**：通过挥动手臂来补偿角动量。
- **环境自适应**：利用膝盖或身体侧面在狭窄空间支撑。
- **坐姿推进（非常规）：** [Stay Seated](../entities/paper-stay-seated.md)（arXiv:2608.28090）在 **被动万向椅** 上学习 G1 **全向坐姿速度跟踪**，脚–地推进 + 非固定骨盆–椅接触，零样本 sim2real；走向 seated loco-manipulation 的第一步。

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
- [CReF（arXiv:2603.29452）](../../sources/papers/cref_arxiv_2603_29452.md) — 单阶段前向深度交叉注意 + 落脚奖励；X2 Ultra 零样本楼梯/台/沟；训练代码未开源。
- [sources/papers/eth-g1-diffusion.md](../../sources/papers/eth-g1-diffusion.md) — 基于扩散模型与 RL 的全身移动框架。
- [sources/papers/humanoid_hardware.md](../../sources/papers/humanoid_hardware.md) — 人形机器人硬件平台综述。
- [QuietWalk（arXiv:2604.23702）](../../sources/papers/quietwalk_arxiv_2604_23702.md) — PINN 估计竖直 GRF + RL 冲击惩罚，G1 跨鞋型低噪行走。
- [GaitSpan（arXiv:2607.12114）](../../sources/papers/gaitspan_arxiv_2607_12114.md) — 行走种子 + GaitWave/H-SLIP 技能生长，单策略连续走–慢跑–跑，五 embodiment 与户外零样本。
- [被动轮轮滑 AMP（arXiv:2607.10815）](../../sources/papers/roller_skating_amp_arxiv_2607_10815.md) — Booster T1 被动轮滑，切片圆柱轮仿真 + 双 gait AMP-PPO，Pump/Push Glide 真机验证。
- [RAVEN（arXiv:2607.15701）](../../sources/papers/raven_rl_adaptive_visibility_graph_arxiv_2607_15701.md) — RL 自适应可见图膨胀 + DAVG-cfMPC + Booster Gym，延迟下人形导航。
- [ADP（arXiv:2607.03454）](../../sources/papers/adp_arxiv_2607_03454.md) — 动力学对抗先验抗扰 locomotion；代码待发布。
- [HumoSlope（arXiv:2607.07830）](../../sources/papers/humoslope_arxiv_2607_07830.md) — 局部平面 ZMP + BSGA 坡条件步态；G1 盲穿户外草地坡 32.1°；代码未开源。
- [P³（arXiv:2607.25541）](../../sources/papers/p3_arxiv_2607_25541.md) — VAE-PPO 边缘似然；G1 踏石/楼梯/缺口真机。
- [ParkourFormer（arXiv:2605.25782）](../../sources/papers/parkourformer_arxiv_2605_25782.md) — query 历史 + 未来两步 AMP 监督；G1 九类地形单策略 93.85%。
- [TRAMP（IEEE RA-L 2026）](../../sources/papers/tramp_vision_assisted_bipedal_locomotion_ieee_lra_2026.md) — 单阶段深度 + 层次特征/MoE + 平地/楼梯地形相关 AMP；SJTU 人形真机坡/楼梯/高台/沟与户外；代码未开源。

## 关联页面
- [人形机器人运控策略的观测输入](../concepts/humanoid-policy-observation-inputs.md) — 主流运控策略输入的五类划分与获取链路
- [人形机器人运控常见奖励函数分类](../concepts/humanoid-policy-reward-functions.md) — 运控 RL 奖励项的六类划分与权重量级
- [Learning Whole-Body Humanoid Locomotion（ETH G1）](../entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) — 扩散运动生成 + RL 全身跟踪，真机箱攀/跨栏/楼梯与混合地形
- [SPRINT 人形竞技冲刺频谱先验](../entities/paper-sprint-humanoid-athletic-sprints.md) — 极少 MoCap + 频域先验外推至高动态冲刺
- [SSR 开放世界人形穿越](../entities/paper-ssr-humanoid-open-world-traversal.md) — 想象落脚点 + 潜空间对称 + 分地形 AMP，楼梯/沟壑/高台与户外长程
- [SOLO 长程感知行走](../entities/paper-solo.md) — QR 逐格高程 + TA-MSE；天工 Omni 单胸深相机零样本 **1.5 km**（未开源）
- [CReF 深度条件融合行走](../entities/paper-cref.md) — 无 2.5D 中间层的单阶段 raw 深度；本体查询注意 + GRU highway + 可支撑落脚奖励；X2 Ultra 零样本
- [Now You See That 端到端视觉人形 locomotion](../entities/paper-now-you-see-that-humanoid-vision-locomotion.md) — 8 步立体深度增广 + 多 critic/discriminator 特权 RL + vision-aware DAgger 蒸馏，双向长楼梯与跑酷零样本
- [QuietWalk 物理感知低噪行走](../entities/paper-quietwalk-humanoid-locomotion.md) — 逆动力学 PINN 估计 GRF 作冲击惩罚；G1 真机 1.2 m/s 降噪约 7 dB，跨赤脚/运动鞋/高跟鞋与多地面材质
- [Learning Quiet Walking（aibo）](../entities/paper-learning-quiet-walking-aibo.md) — 足端接触速度代理的低噪四足对照（同名项目页 QuietWalk；ICRA 2025）
- [GaitSpan 从行走到跑步的技能生长](../entities/paper-gaitspan-humanoid-locomotion-walking-running.md) — 冻结行走种子 + GaitWave 节律组合 + H-SLIP 动态步幅；Booster T1/K1、G1 真机户外走–慢跑–跑连续变速
- [HumoSlope 极端坡面物理引导步态](../entities/paper-humoslope-physics-guided-slope-locomotion.md) — slope-adaptive ZMP + BSGA；G1 盲穿户外草地坡至 32.1°；代码未开源
- [G1 Confined-Space WBP](../entities/paper-g1-confined-space-wbp.md) — 狭窄空间三阶段全身规划 + 残差跟踪；超 NIST 孔洞/倾斜楼梯（arXiv:2608.10220；未开源）
- [P³](../entities/paper-p3.md) — VAE 高程 latent + PPO 边缘似然；G1 踏石/楼梯/缺口真机（arXiv:2607.25541，已开源）
- [WM-LOCO](../entities/paper-wm-loco.md) — RSSM+PPO 单深度预测特征；仿真沟/踏石上匹配 PPO 为 0%，G1 机载三类平均 93.3%（arXiv:2609.02542；代码待发布）
- [Safe-Stop](../entities/paper-safe-stop-humanoid.md) — 急停可停止性双估计 + 阻尼 fallback；G1 OOD 停止 96.4%（arXiv:2609.02358；代码待发布）
- [BRIDGE](../entities/paper-bridge-humanoid.md) — 形态–控制共设计的 80 cm / 12.5 kg / 21 DoF / ~$1500 人形（arXiv:2609.03497；CAD 已放，控制仓待录用）
- [VB-Com](../entities/paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md) — 视觉/盲策略复合，高程图失效时切盲走恢复（G1/H1，ICRA 2026；代码 coming soon）
- [被动轮人形轮滑 AMP（Tsinghua）](../entities/paper-roller-skating-amp-humanoid-passive-wheels.md) — 被动轮滑 + 9 片圆柱碰撞模型；人体 MoCap→GMR→独立 AMP 学 Pump Glide / Push Glide
- [ADP 对抗动力学先验](../entities/paper-adp.md) — SRBD-TO + 动力学窗对抗奖励，推扰相对 AMP 更稳；代码 coming soon
- [RAVEN：RL 自适应可见图 + cf-MPC](../entities/paper-raven-rl-adaptive-visibility-graph-mpc.md) — 导航层 RL 改障碍膨胀，行走层 Booster Gym；延迟与噪声下鲁棒导航
- [Chasing Autonomy Pipeline](../methods/chasing-autonomy-pipeline.md) — 结合重定向与控制引导的 RL 实现高性能奔跑
- [楼梯与障碍感知移动](./stair-obstacle-perceptive-locomotion.md) — 带/不带感知的上下楼梯与越障挂接点
- [Locomotion](./locomotion.md)
- [ZEST](../methods/zest.md) — Boston Dynamics 跨形态高动态模仿与零样本部署
- [MTRG / GfR](../methods/mtrg-reference-goal-driven-rl.md) — RSS 2026；G1 箱式跑酷：参考塑形 + goal 泛化（超越 ZEST tracking 的 OOD 鲁棒性）
- [HIL](../methods/hil-hybrid-imitation-learning.md) / [HIL 论文实体](../entities/paper-hil-hybrid-imitation-learning.md) — TOG 2026 物理角色跑酷 + heading；官方代码未开源
- [HIL vs MTRG vs ZEST 跑酷路线对比](../comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md) — 跑酷模仿三条路线选型
- [Light-Loco-Parkour（LightLP）](../entities/paper-light-loco-parkour.md) — Light Origins / Lightbot 0；稀疏种子 Real2Sim2Real + 多专家蒸馏，无技能标签机载深度跑酷（代码未开源）
- [ParkourFormer](../entities/paper-parkourformer.md) — HKUST-GZ 等；Transformer 查询历史 + 未来两步 AMP 监督；G1 九类地形单策略平均穿越 93.85%（代码 Coming Soon）
- [TRAMP（IEEE RA-L 2026）](../entities/paper-tramp-vision-assisted-bipedal-locomotion.md) — SJTU；单阶段低成本深度 + MoE + 平地/楼梯地形相关 AMP；真机坡/楼梯/高台/宽沟与户外（代码未开源）
- [Diffusion-based Motion Generation](../methods/diffusion-motion-generation.md)
- [PPO](../methods/policy-optimization.md)
- [Whole-Body Coordination](../concepts/whole-body-coordination.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)

## 推荐继续阅读

- [机器人论文阅读笔记：Now You See That](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Now_You_See_That_Learning_End-to-End_Humanoid_Locomotion_from_Raw_Pixels/Now_You_See_That_Learning_End-to-End_Humanoid_Locomotion_from_Raw_Pixels.html)
- [Now You See That 项目页](https://hellod035.github.io/Now_You_See_That/) — RSS 2026；立体深度增广与实机跑酷/楼梯 demo
- [机器人论文阅读笔记：HoRD](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/05_Locomotion/HoRD__Robust_Humanoid_Control_via_History-Conditioned_RL_and_Online_Distillation/HoRD__Robust_Humanoid_Control_via_History-Conditioned_RL_and_Online_Distillation.html)
