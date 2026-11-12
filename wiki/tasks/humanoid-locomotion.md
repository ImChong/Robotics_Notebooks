---
type: task
tags: [humanoid, locomotion, whole-body-control]
status: complete
updated: 2026-11-12
related:
  - ../entities/paper-viola-human-data-control.md
  - ./locomotion.md
  - ./stair-obstacle-perceptive-locomotion.md
  - ../entities/paper-cref.md
  - ../entities/paper-acp-humanoid-mobility.md
  - ../entities/paper-redact-robust-perceptive-locomotion.md
  - ../concepts/humanoid-policy-observation-inputs.md
  - ../concepts/humanoid-policy-reward-functions.md
  - ../concepts/terrain-adaptation.md
  - ../entities/paper-humoslope-physics-guided-slope-locomotion.md
  - ../entities/paper-gaitspan-humanoid-locomotion-walking-running.md
  - ../entities/paper-walk-like-this-stylized-locomotion.md
  - ../entities/paper-humble-human-motion-driven-behavior.md
  - ../entities/paper-roller-skating-amp-humanoid-passive-wheels.md
  - ../entities/paper-now-you-see-that-humanoid-vision-locomotion.md
  - ../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md
  - ../entities/paper-light-loco-parkour.md
  - ../entities/paper-parkourformer.md
  - ../entities/paper-adp.md
  - ../entities/paper-learning-quiet-walking-aibo.md
  - ../entities/paper-quietwalk-humanoid-locomotion.md
  - ../entities/paper-tactilestep.md
  - ../entities/paper-stay-seated.md
  - ../entities/paper-g1-confined-space-wbp.md
  - ../entities/paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md
  - ../entities/paper-p3.md
  - ../entities/paper-evolution-humanoid-locomotion-control.md
  - ../entities/paper-wm-loco.md
  - ../entities/paper-safe-stop-humanoid.md
  - ../entities/paper-bridge-humanoid.md
  - ../entities/paper-gm-loco.md
  - ../entities/paper-robodreamer-anticipatory-humanoid-locomotion.md
  - ../entities/paper-generate-track-improve.md
  - ../entities/paper-anti-gravity-flying-humanoid-wbc-mpc.md
  - ../entities/paper-visible-reachable-workspace-humanoid-design.md
  - ../entities/paper-swap-humanoid-collision-avoidance.md
original:
  - ../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md
summary: "人形机器人在复杂地形下的平衡与移动任务，强调高维动力学处理、环境感知以及全身肢体协调。"
---

# Humanoid Locomotion (人形机器人移动)

**Humanoid Locomotion**：使双足类人机器人能够在复杂、非结构化的地形中，保持平衡的同时实现高效、鲁棒的位移，并具备全身协调（Whole-body Coordination）能力。

## 一句话定义

让两条腿（甚至加上手和膝盖）在各种烂路上走稳、走远、走得像人。

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

## 为什么重要

- 人形是通用机器人的主要载体：Radosavovic 等（arXiv:2303.03381）开篇即指出，通用任务需要通用本体与通用控制器，而经典方法难以泛化到新环境。
- locomotion 是人形一切下游能力（操作、导航、作业）的前提；本页下文各论文均以“站稳、走远”为共同底座。
- 学习式控制已能在全尺寸人形上零样本上真机：该论文在 Digit 上一周户外测试无跌倒（原文结果）。

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
- **坡面与交互力联合跟踪：** [BRACE](../entities/paper-brace-force-terrain-whole-body-tracking.md) 将平地参考按 terrain 与 commanded wrench 适配，再以 DAgger 把施力/补偿教师蒸馏为 proprioception 驱动的全身 flow policy。
- **坐姿推进（非常规）：** [Stay Seated](../entities/paper-stay-seated.md)（arXiv:2608.28090）在 **被动万向椅** 上学习 G1 **全向坐姿速度跟踪**，脚–地推进 + 非固定骨盆–椅接触，零样本 sim2real；走向 seated loco-manipulation 的第一步。

## 按主题索引：论文与方法

> 站内与人形移动相关的论文、方法与路线对比按主题归档；同一页面只在最贴切的一组出现一次。核心概念与方法见文末「关联页面」。

### 感知行走与复杂地形

- [SSR 开放世界人形穿越](../entities/paper-ssr-humanoid-open-world-traversal.md) — 想象落脚点 + 潜空间对称 + 分地形 AMP，楼梯/沟壑/高台与户外长程
- [ACP：注意力式接触规划人形移动](../entities/paper-acp-humanoid-mobility.md) — 用接触规划与条件跟踪器连接动作先验和强化学习地形适应（arXiv:2610.31608）
- [SOLO 长程感知行走](../entities/paper-solo.md) — QR 逐格高程 + TA-MSE；天工 Omni 单胸深相机零样本 **1.5 km**（未开源）
- [FootQuery 深度历史触地检索](../entities/paper-footquery-perceptive-humanoid-locomotion.md) — per-foot 下一触地点 query 深度历史；G1 户外/室内楼梯·平台·沟；arXiv:2609.21447（无官方代码）
- [CReF 深度条件融合行走](../entities/paper-cref.md) — 无 2.5D 中间层的单阶段 raw 深度；本体查询注意 + GRU highway + 可支撑落脚奖励；X2 Ultra 零样本
- [Now You See That 端到端视觉人形 locomotion](../entities/paper-now-you-see-that-humanoid-vision-locomotion.md) — 8 步立体深度增广 + 多 critic/discriminator 特权 RL + vision-aware DAgger 蒸馏，双向长楼梯与跑酷零样本
- [TRAMP（IEEE RA-L 2026）](../entities/paper-tramp-vision-assisted-bipedal-locomotion.md) — SJTU；单阶段低成本深度 + MoE + 平地/楼梯地形相关 AMP；真机坡/楼梯/高台/宽沟与户外（代码未开源）
- [VB-Com](../entities/paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md) — 视觉/盲策略复合，高程图失效时切盲走恢复（G1/H1，ICRA 2026；代码 coming soon）
- [P³](../entities/paper-p3.md) — VAE 高程 latent + PPO 边缘似然；G1 踏石/楼梯/缺口真机（arXiv:2607.25541，已开源）
- [REDACT](../entities/paper-redact-robust-perceptive-locomotion.md) — clean-only 仿真训练 + 持续特征遮蔽 + 干净观测共识门控；项目页含编码器和校准参数，算法代码未公开（arXiv:2609.25450）
- [WM-LOCO](../entities/paper-wm-loco.md) — RSSM+PPO 单深度预测特征；仿真沟/踏石上匹配 PPO 为 0%，G1 机载三类平均 93.3%（arXiv:2609.02542；代码待发布）
- [DWMP](../entities/paper-dwmp.md) — Koopman 本体 WM + DepthDreamer 深度 WM 双路表征；MuJoCo 五类越障 SR 0.85–0.95，G1 真机随机布局最高 0.95（arXiv:2609.12347；未开源）
- [RoboDreamer](../entities/paper-robodreamer-anticipatory-humanoid-locomotion.md) — Mamba PSSM 前瞻行走 + 推理动作细化（arXiv:2609.07096；未见代码）
- [GM-Loco 颗粒介质地形自适应](../entities/paper-gm-loco.md) — 3D RFT 颗粒接触 + VAE Teacher-Student；G1 玄武岩/干沙/海滩沙走跑至 2.5 m/s（arXiv:2609.10286；代码待发布）
- [HumoSlope 极端坡面物理引导步态](../entities/paper-humoslope-physics-guided-slope-locomotion.md) — slope-adaptive ZMP + BSGA；G1 盲穿户外草地坡至 32.1°；代码未开源
- [G1 屋顶斜坡全身作业 Locomotion](../entities/paper-g1-slope-adaptive-roofing-locomotion.md) — PICO 演示→metric 坡面 mesh 轨迹优化→相位门控 RL；钉枪/锤/推 clearance 0.26–0.53 cm；UF/G1（arXiv:2609.20558；未开源）
- [G1 Confined-Space WBP](../entities/paper-g1-confined-space-wbp.md) — 狭窄空间三阶段全身规划 + 残差跟踪；超 NIST 孔洞/倾斜楼梯（arXiv:2608.10220；未开源）

### 复杂空间避障

- [SWAP：测试时可控的人形全身避障](../entities/paper-swap-humanoid-collision-avoidance.md) — 感知条件化扩散动作提案 + 低层 RL 稳定跟踪；仿真与 Unitree G1 静态/动态障碍，代码待开放（arXiv:2610.25711）

### 跑酷、冲刺与高动态技能

- [Push the Limit：高速冲刺与激进转弯](../entities/paper-push-the-limit-humanoid-sprinting.md) — 因果 MHA 历史编码 + 航向方向奖励 + 技能条件 Multi-AMP；减重真机 100 m 平均 7.14 m/s（CoRL 2026）
- [SPRINT 人形竞技冲刺频谱先验](../entities/paper-sprint-humanoid-athletic-sprints.md) — 极少 MoCap + 频域先验外推至高动态冲刺
- [GaitSpan 从行走到跑步的技能生长](../entities/paper-gaitspan-humanoid-locomotion-walking-running.md) — 冻结行走种子 + GaitWave 节律组合 + H-SLIP 动态步幅；Booster T1/K1、G1 真机户外走–慢跑–跑连续变速
- [Chasing Autonomy Pipeline](../methods/chasing-autonomy-pipeline.md) — 结合重定向与控制引导的 RL 实现高性能奔跑
- [ZEST](../methods/zest.md) — Boston Dynamics 跨形态高动态模仿与零样本部署
- [MTRG / GfR](../methods/mtrg-reference-goal-driven-rl.md) — RSS 2026；G1 箱式跑酷：参考塑形 + goal 泛化（超越 ZEST tracking 的 OOD 鲁棒性）
- [HIL](../methods/hil-hybrid-imitation-learning.md) / [HIL 论文实体](../entities/paper-hil-hybrid-imitation-learning.md) — TOG 2026 物理角色跑酷 + heading；官方代码未开源
- [HIL vs MTRG vs ZEST 跑酷路线对比](../comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md) — 跑酷模仿三条路线选型
- [Light-Loco-Parkour（LightLP）](../entities/paper-light-loco-parkour.md) — Light Origins / Lightbot 0；稀疏种子 Real2Sim2Real + 多专家蒸馏，无技能标签机载深度跑酷（代码未开源）
- [ParkourFormer](../entities/paper-parkourformer.md) — HKUST-GZ 等；Transformer 查询历史 + 未来两步 AMP 监督；G1 九类地形单策略平均穿越 93.85%（代码 Coming Soon）

### 运动先验与生成式控制

- [Walk Like This](../entities/paper-walk-like-this-stylized-locomotion.md) — 示例动作编码风格 + 自回归扩散生成 + PPO 跟踪；Unitree G1 / Jetson Thor 实时演示（CoRL 2026；未见可运行代码）
- [Learning Whole-Body Humanoid Locomotion（ETH G1）](../entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) — 扩散运动生成 + RL 全身跟踪，真机箱攀/跨栏/楼梯与混合地形
- [Generate, Track, Improve（GTI）](../entities/paper-generate-track-improve.md) — 双深度 flow matching 生成 + CLF-RL 跟踪 + AWR 微调生成器；G1 户外走跑跳箱/楼梯（arXiv:2609.31577；代码待发布）
- [ADP 对抗动力学先验](../entities/paper-adp.md) — SRBD-TO + 动力学窗对抗奖励，推扰相对 AMP 更稳；代码 coming soon
- [被动轮人形轮滑 AMP（Tsinghua）](../entities/paper-roller-skating-amp-humanoid-passive-wheels.md) — 被动轮滑 + 9 片圆柱碰撞模型；人体 MoCap→GMR→独立 AMP 学 Pump Glide / Push Glide

### 低噪、触觉与安全停止

- [QuietWalk 物理感知低噪行走](../entities/paper-quietwalk-humanoid-locomotion.md) — 逆动力学 PINN 估计 GRF 作冲击惩罚；G1 真机 1.2 m/s 降噪约 7 dB，跨赤脚/运动鞋/高跟鞋与多地面材质
- [TactileStep 足底触觉跑酷](../entities/paper-tactilestep.md) — 压力鞋垫特征对齐仿真 + 四相位软着陆/稳支撑奖励；相对 Hiking 基线真机冲击与噪声显著下降（CoRL 2026 Spotlight；代码待发布）
- [Learning Quiet Walking（aibo）](../entities/paper-learning-quiet-walking-aibo.md) — 足端接触速度代理的低噪四足对照（同名项目页 QuietWalk；ICRA 2025）
- [Safe-Stop](../entities/paper-safe-stop-humanoid.md) — 急停可停止性双估计 + 阻尼 fallback；G1 OOD 停止 96.4%（arXiv:2609.02358；代码待发布）

### 硬件与标定

- [可见—可达工作空间硬件设计](../entities/paper-visible-reachable-workspace-humanoid-design.md) — Duke V2 双 RGB-D；覆盖率 38%→97%（arXiv:2609.08905；未见代码）
- [BRIDGE](../entities/paper-bridge-humanoid.md) — 形态–控制共设计的 80 cm / 12.5 kg / 21 DoF / ~$1500 人形（arXiv:2609.03497；CAD 已放，控制仓待录用）
- [下肢关节零位标定](../entities/paper-contact-constrained-joint-offset-calibration.md) — 双支撑脚间变换一致性；AgiBot A2/A3（arXiv:2609.02306；未开源）
- [双凸轮并联弹性踝](../entities/paper-dual-cam-parallel-elastic-ankle.md) — IIT 2-DoF PEA 单气弹簧双轴补偿（arXiv:2608.30832）

### 相邻方向：飞行人形、导航、交互与移动操作

- [飞行人形反重力行走](../entities/paper-anti-gravity-flying-humanoid-wbc-mpc.md) — 推力率输入 WBC-MPC；东大（arXiv:2609.07544；未见代码）
- [RAVEN：RL 自适应可见图 + cf-MPC](../entities/paper-raven-rl-adaptive-visibility-graph-mpc.md) — 导航层 RL 改障碍膨胀，行走层 Booster Gym；延迟与噪声下鲁棒导航
- [IHMC 可编辑 loco-manipulation 系统](../entities/paper-ihmc-fast-resilient-loco-manipulation.md) — Affordance Template + 行为树；H1-2/Alex 推门 34 s（arXiv:2609.01518；未见代码）
- [RoboGesture](../entities/paper-robogesture.md) — 流式语音→G1 语义手势；DiT-CFM + MPC 安全滤波（arXiv:2608.28693；未开源）
- [动态多模态 HRI 数据集协议](../entities/paper-dynamic-multimodal-hri-dataset-u1.md) — Unitree U1 三档复杂度采集设计（arXiv:2609.03255；数据待发布）

## 核心原理

> 以下以 Radosavovic 等（arXiv:2303.03381，已通读）为代表说明“仿真大规模 RL + 零样本上真机”这一主流范式；与上文各路线的关系为本页归纳。

- **POMDP 表述**：真实机器人无法获得真实状态，策略只能依据带噪观测，故按部分可观测 MDP 处理，用观测-动作历史补足信息。
- **教师-学生联合训练**：先用特权状态训练教师策略，再训练只看观测的学生，目标 L = L_RL + λ·KL(π_o‖π_s)，λ 退火到零，使学生先借力教师、后超越之。
- **历史即上下文**：因果 Transformer 读取最近 16 步观测-动作对；步态随地形变化、足部卡阻后抬腿更高等行为为涌现，并非预设。
- **大规模并行仿真 + 域随机化**：随机化动力学、控制参数、地形、观测噪声与延迟，再零样本部署；先在厂商高保真仿真器筛选策略。

## 工程实践

- 观测与奖励的具体设计见 [观测输入](../concepts/humanoid-policy-observation-inputs.md) 与 [奖励函数](../concepts/humanoid-policy-reward-functions.md)。
- 闭链/欠驱动结构（如 Digit 小腿-跗骨）若仿真器不支持，原文用高刚度虚拟弹簧近似，并强调仍是 sim-to-real 的额外难点。
- 部署频率上，原文策略 50 Hz、关节 PD 1 kHz；盲策略遇台阶会磕绊，需感知时参见 [楼梯与障碍感知移动](./stair-obstacle-perceptive-locomotion.md)。
- 消融结论（原文）：Transformer 优于 MLP/TCN/LSTM，更长上下文更好，联合 KL+RL 优于单一目标。

## 局限

- 代表工作（Digit，纯本体感知）自述：左右步态不完全对称、速度跟踪不完美、极强外扰仍会摔倒；盲策略对离散障碍只能事后适应。
- 仿真器对闭链结构只能近似，影响迁移上限。
- 本页多数条目为单篇论文的自报结果，平台（G1、Digit、T1 等）与评测协议不统一，横向数字不宜直接比较（本页归纳）。

## 参考来源
- [Real-World Humanoid Locomotion with RL（arXiv:2303.03381，通读归档）](../../sources/papers/humanoid_locomotion_reading_arxiv_2303_03381.md) — 因果 Transformer + 教师 KL 蒸馏/RL 联合，Digit 零样本户外行走。
- [Walk Like This（CoRL 2026）](../../sources/papers/walk_like_this_corl2026.md) — 示例条件风格化 locomotion 与 G1 实机追踪。
- [Chasing Autonomy: Dynamic Retargeting and Control Guided RL for Performant and Controllable Humanoid Running](../../sources/papers/chasing_autonomy.md)
- [SPRINT: Efficient Spectral Priors for Humanoid Athletic Sprints](../../sources/papers/sprint_arxiv_2605_28549.md) — 5 条参考 + 频谱先验 + 残差 RL，G1 真机冲刺 6 m/s。
- [Push the Limit（CoRL 2026）](../../sources/papers/push-the-limit-humanoid-sprinting-arxiv-2610-14840.md) — 时序注意力、航向方向奖励与技能条件 Multi-AMP，解决高速冲刺中的转弯和多技能风格监督。
- [SSR: Scaling Surefooted and Symmetric Humanoid Traversal to the Open World](../../sources/papers/ssr_arxiv_2605_30770.md) — 第一视角深度单阶段 PPO + 想象落脚点，AgiBot X2 户外 1.3 km 长程。
- [CReF（arXiv:2603.29452）](../../sources/papers/cref_arxiv_2603_29452.md) — 单阶段前向深度交叉注意 + 落脚奖励；X2 Ultra 零样本楼梯/台/沟；训练代码未开源。
- [sources/papers/eth-g1-diffusion.md](../../sources/papers/eth-g1-diffusion.md) — 基于扩散模型与 RL 的全身移动框架。
- [sources/papers/humanoid_hardware.md](../../sources/papers/humanoid_hardware.md) — 人形机器人硬件平台综述。
- [QuietWalk（arXiv:2604.23702）](../../sources/papers/quietwalk_arxiv_2604_23702.md) — PINN 估计竖直 GRF + RL 冲击惩罚，G1 跨鞋型低噪行走。
- [TactileStep（arXiv:2609.28959）](../../sources/papers/tactilestep_arxiv_2609_28959.md) — 足底 sole 特征进 actor + 相位触觉奖励；G1 跑酷真机；以 Hiking 为强基线。
- [GaitSpan（arXiv:2607.12114）](../../sources/papers/gaitspan_arxiv_2607_12114.md) — 行走种子 + GaitWave/H-SLIP 技能生长，单策略连续走–慢跑–跑，五 embodiment 与户外零样本。
- [被动轮轮滑 AMP（arXiv:2607.10815）](../../sources/papers/roller_skating_amp_arxiv_2607_10815.md) — Booster T1 被动轮滑，切片圆柱轮仿真 + 双 gait AMP-PPO，Pump/Push Glide 真机验证。
- [RAVEN（arXiv:2607.15701）](../../sources/papers/raven_rl_adaptive_visibility_graph_arxiv_2607_15701.md) — RL 自适应可见图膨胀 + DAVG-cfMPC + Booster Gym，延迟下人形导航。
- [ADP（arXiv:2607.03454）](../../sources/papers/adp_arxiv_2607_03454.md) — 动力学对抗先验抗扰 locomotion；代码待发布。
- [HumoSlope（arXiv:2607.07830）](../../sources/papers/humoslope_arxiv_2607_07830.md) — 局部平面 ZMP + BSGA 坡条件步态；G1 盲穿户外草地坡 32.1°；代码未开源。
- [P³（arXiv:2607.25541）](../../sources/papers/p3_arxiv_2607_25541.md) — VAE-PPO 边缘似然；G1 踏石/楼梯/缺口真机。
- [ParkourFormer（arXiv:2605.25782）](../../sources/papers/parkourformer_arxiv_2605_25782.md) — query 历史 + 未来两步 AMP 监督；G1 九类地形单策略 93.85%。
- [TRAMP（IEEE RA-L 2026）](../../sources/papers/tramp_vision_assisted_bipedal_locomotion_ieee_lra_2026.md) — 单阶段深度 + 层次特征/MoE + 平地/楼梯地形相关 AMP；SJTU 人形真机坡/楼梯/高台/沟与户外；代码未开源。


### 人类动作数据驱动的通用控制

- [VioLA：从人类数据学习人形通用控制](../entities/paper-viola-human-data-control.md) — 以共享运动潜空间复用人类动作数据，并报告 G1 真机移动与操作结果。

## 关联页面

- [Locomotion](./locomotion.md)
- [楼梯与障碍感知移动](./stair-obstacle-perceptive-locomotion.md) — 带/不带感知的上下楼梯与越障挂接点
- [人形机器人运控策略的观测输入](../concepts/humanoid-policy-observation-inputs.md) — 主流运控策略输入的五类划分与获取链路
- [人形机器人运控常见奖励函数分类](../concepts/humanoid-policy-reward-functions.md) — 运控 RL 奖励项的六类划分与权重量级
- [PPO](../methods/policy-optimization.md)
- [Diffusion-based Motion Generation](../methods/diffusion-motion-generation.md)
- [Whole-Body Coordination](../concepts/whole-body-coordination.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)

## 推荐继续阅读

- [Evolution of Humanoid Locomotion Control（Science Robotics 2026 Review）](../entities/paper-evolution-humanoid-locomotion-control.md) — 六十年 locomotion 控制三时代 + 280 refs companion
- [机器人论文阅读笔记：Now You See That](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Now_You_See_That_Learning_End-to-End_Humanoid_Locomotion_from_Raw_Pixels/Now_You_See_That_Learning_End-to-End_Humanoid_Locomotion_from_Raw_Pixels.html)
- [Now You See That 项目页](https://hellod035.github.io/Now_You_See_That/) — RSS 2026；立体深度增广与实机跑酷/楼梯 demo
- [机器人论文阅读笔记：HoRD](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/05_Locomotion/HoRD__Robust_Humanoid_Control_via_History-Conditioned_RL_and_Online_Distillation/HoRD__Robust_Humanoid_Control_via_History-Conditioned_RL_and_Online_Distillation.html)

## 持续动作学习实例

- [Continual Humanoid Motion Learning](../entities/paper-continual-humanoid-motion-learning.md) — similarity-guided LoRA-PNN 按动作相似度扩展 G1 全身动作策略，并以旧列冻结降低遗忘（arXiv:2610.04231）。

## 人体步态先验与可转向策略

- [HuMBLE：人体动作驱动的具身 locomotion](../entities/paper-humble-human-motion-driven-behavior.md) — 人体动捕→全身 teacher 蒸馏→参考模仿 + 指令跟踪双任务 PPO；在 Atlas R1/D1 与 Unitree G1 验证（arXiv:2610.10489）。


## 新进展：非对称负载与动作局部编辑

- [ReACT：单手负载速度跟踪](../entities/paper-react-asymmetric-load-tracking.md)（arXiv:2611.13929）以反射一致的外载调制、跨负载轨迹一致性训练提高 G1 单手持物时的移动稳定性；真机侧向误差并非所有项目领先。
- [RedirectMimic：局部动作内容抑制](../entities/paper-redirectmimic.md)（arXiv:2611.14049）对 G1 模仿策略训练期的上肢局部奖励参考做编辑，在不编辑其他身体段和未编辑时间窗的情况下抑制指定动作。它解决动作内容约束，不等同于 ReACT 的负载速度跟踪。
