# 从域随机化到残差学习：Sim2Real 技术路线梳理

> 来源归档（blog / 微信公众号）

- **标题：** 从域随机化到残差学习：Sim2Real 技术路线梳理
- **类型：** blog
- **作者：** 自由度FreeDof（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/K_6MibGXWwh9OL9eSZxOMg
- **发表日期：** 2026-08-23
- **入库日期：** 2026-09-11
- **抓取方式：** WebFetch（桌面 UA 返回微信验证页；正文由 WebFetch 可读通道获取）
- **原始抓取落盘：** [`sources/raw/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md`](../raw/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- **一句话说明：** 以 **可辨识性** 为轴，把 Sim2Real 四条主流路线（系统辨识 / 域随机化 / 在线适应 / 残差学习）放在同一问题下比较；给出分层组合、症状查表与论文阅读判据。
- **开源状态（步骤 2.5）：** 综述文，无项目页、无代码仓 → **步骤 2.5 不适用**。

## 核心摘录（归纳，非全文）

### 核心问题

仿真里优化 $J_{\mathrm{sim}}(\pi)$，部署却要 $J_{\mathrm{real}}(\pi)$ 高——四条路线都在缩小这个差。共同问题：**真机动力学参数能否辨识？剩余误差如何处理？**

### 四条路线（立场对照）

| 路线 | 对可辨识性的立场 | 真机数据 | 典型代价 |
|------|------------------|----------|----------|
| **系统辨识** | 正面求解 | 专门辨识实验 | 参数须可辨、实验设计要求高 |
| **域随机化** | 放弃辨识，用覆盖换鲁棒 | 零（训练期） | 保守，敏捷上限受限 |
| **在线适应** | 推迟到运行时，只要求控制相关上下文可区分 | 部署期历史 | 激励不足时**静默退化**为 DR |
| **残差学习** | 不强求完整参数辨识，直接拟合修正项 | 真机 rollout | 只在训练分布内有效 |

### 分层组合（成熟系统读法）

先把能辨的辨出来（缩小不确定性）→ 对剩余不确定性做**窄 DR** → 再处理辨不出的部分（残差 / 适应）。顺序不能反。

### 文内点名代表作（→ 本库节点）

| 主题 | 代表 | wiki |
|------|------|------|
| 执行器 SysID + 零样本 | PACE | [paper-pace-sim2real-legged-robots](../../wiki/entities/paper-pace-sim2real-legged-robots.md) |
| 主动激励实验设计 | SPI-Active | [paper-notebook-sampling-based-system-identification-with-active](../../wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md) |
| 执行器残差 / UAN | Fey et al. 2025 | [actuator-network](../../wiki/methods/actuator-network.md) |
| 动作层残差 | ASAP | [paper-hrl-stack-25-asap](../../wiki/entities/paper-hrl-stack-25-asap.md) |
| 在线适应 | RMA / UP-OSI | [paper-rma-rapid-motor-adaptation](../../wiki/entities/paper-rma-rapid-motor-adaptation.md) |
| 多引擎 DR | PolySim | [paper-polysim-multi-simulator-humanoid-sim2real](../../wiki/entities/paper-polysim-multi-simulator-humanoid-sim2real.md) |
| 部署监控 | RAPT | [paper-rapt-sim2real-ood-detection](../../wiki/entities/paper-rapt-sim2real-ood-detection.md) |
| 单关节实验设计深读 | 姊妹篇 | [sim2real-joint-sysid-experiment-design](../../wiki/methods/sim2real-joint-sysid-experiment-design.md) |

### 症状查表（节选）

| 症状 | 优先方向 |
|------|----------|
| 固定基座关节响应对不上 | 时间同步 / 单位 → 闭环执行器辨识（勿先调 PPO） |
| 回差、迟滞、柔性明显 | 可解释主效应 + 力矩层残差 |
| 固定基座吻合、落地失败 | 接触参数 / 状态估计 / 时延；需含接触辨识 |
| 敏捷动作跟不上 | 名义模型做准 + 动作层残差；步频是 reality gap 敏感代理 |
| 完全没有真机数据 | 宽 DR / ADR / 教师–学生 / 多引擎；仍需真机验收 |

## 对 wiki 的映射

- **对比页：** [sim2real-four-routes-identifiability](../../wiki/comparisons/sim2real-four-routes-identifiability.md)
- **44 篇地图：** [freedof-sim2real-44-papers-technology-map](../../wiki/overview/freedof-sim2real-44-papers-technology-map.md)
- **44 篇 catalog：** [freedof_sim2real_44_catalog.md](../papers/freedof_sim2real_44_catalog.md)
- **姊妹篇（已入库）：** [wechat_freedof_sim2real_dynamics_identification.md](./wechat_freedof_sim2real_dynamics_identification.md) → [关节动力学辨识实验设计](../../wiki/methods/sim2real-joint-sysid-experiment-design.md)
- 交叉：[Sim2Real](../../wiki/concepts/sim2real.md)、[闭环误差分层工程](../../wiki/queries/sim2real-closed-loop-engineering.md)、[Sim2Real Approaches](../../wiki/comparisons/sim2real-approaches.md)

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 升格对比页（四条路线 + 可辨识性轴）
- [x] 与姊妹篇 SysID 实验设计页交叉链接
- [x] 44 篇参考文献独立详情节点（0 重复 arXiv）

### 44 篇参考文献 → 独立详情节点（2026-09-20 补齐）

| # | 章节 | 论文 | arXiv | 节点 | wiki |
|---|------|------|-------|------|------|
| 01 | 系统辨识 | Parameter identification of robot dynamics | — | **新建** | [paper-khosla-robot-dynamics-parameter-identification-1985](../../wiki/entities/paper-khosla-robot-dynamics-parameter-identification-1985.md) |
| 02 | 系统辨识 | On the identification of the inertial parameters of rob | — | **新建** | [paper-gautier-khalil-inertial-parameter-identification-1988](../../wiki/entities/paper-gautier-khalil-inertial-parameter-identification-1988.md) |
| 03 | 系统辨识 | Towards bridging the gap: Systematic sim-to-real transf | [2509.06342](https://arxiv.org/abs/2509.06342) | **复用** | [paper-pace-sim2real-legged-robots](../../wiki/entities/paper-pace-sim2real-legged-robots.md) |
| 04 | 系统辨识 | Impact of static friction on Sim2Real in robotic reinfo | [2503.01255](https://arxiv.org/abs/2503.01255) | **复用** | [paper-sa-2503-01255-impact-of-static-friction-on-sim2real-in-robotic](../../wiki/entities/paper-sa-2503-01255-impact-of-static-friction-on-sim2real-in-robotic.md) |
| 05 | 系统辨识 | Sampling-based system identification with active explor | [2505.14266](https://arxiv.org/abs/2505.14266) | **复用** | [paper-notebook-sampling-based-system-identification-with-active](../../wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md) |
| 06 | 系统辨识 | Identification and the information matrix: how to get j | — | **新建** | [paper-gevers-identification-information-matrix-2009](../../wiki/entities/paper-gevers-identification-information-matrix-2009.md) |
| 07 | 系统辨识 | Achieving precise and reliable locomotion with differen | [2508.04696](https://arxiv.org/abs/2508.04696) | **新建** | [paper-kovalev-differentiable-simulation-locomotion-sysid](../../wiki/entities/paper-kovalev-differentiable-simulation-locomotion-sysid.md) |
| 08 | 系统辨识 | Simulator adaptation for sim-to-real learning of legged | [2604.11090](https://arxiv.org/abs/2604.11090) | **复用** | [paper-notebook-simulator-adaptation-via-proprioceptive-distribu](../../wiki/entities/paper-notebook-simulator-adaptation-via-proprioceptive-distribu.md) |
| 09 | 域随机化 | Domain randomization for transferring deep neural netwo | [1703.06907](https://arxiv.org/abs/1703.06907) | **复用** | [paper-notebook-domain-randomization-for-transferring-deep-neura](../../wiki/entities/paper-notebook-domain-randomization-for-transferring-deep-neura.md) |
| 10 | 域随机化 | Sim-to-real transfer of robotic control with dynamics r | [1710.06537](https://arxiv.org/abs/1710.06537) | **新建** | [paper-peng-dynamics-randomization-sim2real](../../wiki/entities/paper-peng-dynamics-randomization-sim2real.md) |
| 11 | 域随机化 | Sim-to-real: learning agile locomotion for quadruped ro | [1804.10332](https://arxiv.org/abs/1804.10332) | **新建** | [paper-tan-quadruped-agile-locomotion-sim2real](../../wiki/entities/paper-tan-quadruped-agile-locomotion-sim2real.md) |
| 12 | 域随机化 | Learning dexterous in-hand manipulation | [1808.00177](https://arxiv.org/abs/1808.00177) | **复用** | [paper-pai-1808-00177-learningdexterousinhandmanipulat](../../wiki/entities/paper-pai-1808-00177-learningdexterousinhandmanipulat.md) |
| 13 | 域随机化 | Closing the sim-to-real loop: adapting simulation rando | [1810.05687](https://arxiv.org/abs/1810.05687) | **复用** | [paper-pai-1910-13325-simopt](../../wiki/entities/paper-pai-1910-13325-simopt.md) |
| 14 | 域随机化 | BayesSim: adaptive domain randomization via probabilist | [1906.01728](https://arxiv.org/abs/1906.01728) | **复用** | [paper-pai-1906-01728-bayessim](../../wiki/entities/paper-pai-1906-01728-bayessim.md) |
| 15 | 域随机化 | Data-efficient domain randomization with Bayesian optim | [2003.02471](https://arxiv.org/abs/2003.02471) | **新建** | [paper-muratore-bayesian-optimization-domain-randomization](../../wiki/entities/paper-muratore-bayesian-optimization-domain-randomization.md) |
| 16 | 域随机化 | Solving Rubik's cube with a robot hand (ADR) | [1910.07113](https://arxiv.org/abs/1910.07113) | **复用** | [paper-as-1910-07113-solving-rubik-s-cube-with-a-robot-hand](../../wiki/entities/paper-as-1910-07113-solving-rubik-s-cube-with-a-robot-hand.md) |
| 17 | 域随机化 | EPOpt: learning robust neural network policies using mo | [1610.01283](https://arxiv.org/abs/1610.01283) | **新建** | [paper-epopt-robust-policies-model-ensembles](../../wiki/entities/paper-epopt-robust-policies-model-ensembles.md) |
| 18 | 域随机化 | Robust adversarial reinforcement learning (RARL) | [1703.02702](https://arxiv.org/abs/1703.02702) | **新建** | [paper-rarl-robust-adversarial-rl](../../wiki/entities/paper-rarl-robust-adversarial-rl.md) |
| 19 | 域随机化 | PolySim: bridging the sim-to-real gap for humanoid cont | [2510.01708](https://arxiv.org/abs/2510.01708) | **新建** | [paper-polysim-multi-simulator-humanoid-sim2real](../../wiki/entities/paper-polysim-multi-simulator-humanoid-sim2real.md) |
| 20 | 域随机化 | Simulation tools for model-based robotics: comparison o | — | **新建** | [paper-erez-simulation-tools-comparison-icra-2015](../../wiki/entities/paper-erez-simulation-tools-comparison-icra-2015.md) |
| 21 | 域随机化 | Validating robotics simulators on real-world impacts | [2110.00541](https://arxiv.org/abs/2110.00541) | **新建** | [paper-acosta-validating-simulators-real-world-impacts](../../wiki/entities/paper-acosta-validating-simulators-real-world-impacts.md) |
| 22 | 域随机化 | Contact models in robotics: a comparative analysis | [2304.06372](https://arxiv.org/abs/2304.06372) | **新建** | [paper-le-lidec-contact-models-comparative-analysis](../../wiki/entities/paper-le-lidec-contact-models-comparative-analysis.md) |
| 23 | 在线适应 | Preparing for the unknown: learning a universal policy  | [1702.02453](https://arxiv.org/abs/1702.02453) | **新建** | [paper-up-osi-universal-policy-online-sysid](../../wiki/entities/paper-up-osi-universal-policy-online-sysid.md) |
| 24 | 在线适应 | RMA: rapid motor adaptation for legged robots | [2107.04034](https://arxiv.org/abs/2107.04034) | **复用** | [paper-rma-rapid-motor-adaptation](../../wiki/entities/paper-rma-rapid-motor-adaptation.md) |
| 25 | 在线适应 | Rapid locomotion via reinforcement learning | [2205.02824](https://arxiv.org/abs/2205.02824) | **复用** | [paper-rapid-locomotion-rl](../../wiki/entities/paper-rapid-locomotion-rl.md) |
| 26 | 在线适应 | Real-world humanoid locomotion with reinforcement learn | [2303.03381](https://arxiv.org/abs/2303.03381) | **复用** | [paper-digit-humanoid-locomotion-rl](../../wiki/entities/paper-digit-humanoid-locomotion-rl.md) |
| 27 | 在线适应 | Learning quadrupedal locomotion over challenging terrai | [2010.11251](https://arxiv.org/abs/2010.11251) | **复用** | [paper-notebook-learning-quadrupedal-locomotion-over-challenging](../../wiki/entities/paper-notebook-learning-quadrupedal-locomotion-over-challenging.md) |
| 28 | 在线适应 | Learning to walk in minutes using massively parallel de | [2109.11978](https://arxiv.org/abs/2109.11978) | **复用** | [paper-anymal-walk-minutes-parallel-drl](../../wiki/entities/paper-anymal-walk-minutes-parallel-drl.md) |
| 29 | 残差学习 | Learning agile and dynamic motor skills for legged robo | [1901.08652](https://arxiv.org/abs/1901.08652) | **复用** | [paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg](../../wiki/entities/paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg.md) |
| 30 | 残差学习 | Sim-to-real transfer with neural-augmented robot simula | — | **新建** | [paper-golemo-neural-augmented-robot-simulation](../../wiki/entities/paper-golemo-neural-augmented-robot-simulation.md) |
| 31 | 残差学习 | Bridging the sim-to-real gap for athletic loco-manipula | [2502.10894](https://arxiv.org/abs/2502.10894) | **复用** | [paper-notebook-bridging-the-sim-to-real-gap-for-athletic-loco-m](../../wiki/entities/paper-notebook-bridging-the-sim-to-real-gap-for-athletic-loco-m.md) |
| 32 | 残差学习 | Residual reinforcement learning for robot control | [1812.03201](https://arxiv.org/abs/1812.03201) | **复用** | [paper-residual-rl-robot-control](../../wiki/entities/paper-residual-rl-robot-control.md) |
| 33 | 残差学习 | ASAP: aligning simulation and real-world physics for le | [2502.01143](https://arxiv.org/abs/2502.01143) | **复用** | [paper-hrl-stack-25-asap](../../wiki/entities/paper-hrl-stack-25-asap.md) |
| 34 | 残差学习 | MOSAIC: bridging the sim-to-real gap in generalist huma | [2602.08594](https://arxiv.org/abs/2602.08594) | **复用** | [paper-loco-manip-161-014-mosaic](../../wiki/entities/paper-loco-manip-161-014-mosaic.md) |
| 35 | 残差学习 | Off-dynamics reinforcement learning: training for trans | [2006.13916](https://arxiv.org/abs/2006.13916) | **新建** | [paper-eysenbach-off-dynamics-rl](../../wiki/entities/paper-eysenbach-off-dynamics-rl.md) |
| 36 | 残差学习 | Legged robots that keep on learning: fine-tuning locomo | [2110.05457](https://arxiv.org/abs/2110.05457) | **新建** | [paper-smith-legged-robots-keep-learning](../../wiki/entities/paper-smith-legged-robots-keep-learning.md) |
| 37 | 监控与评测 | RAPT: model-predictive out-of-distribution detection an | [2602.01515](https://arxiv.org/abs/2602.01515) | **新建** | [paper-rapt-sim2real-ood-detection](../../wiki/entities/paper-rapt-sim2real-ood-detection.md) |
| 38 | 监控与评测 | Sim2Real predictivity: does evaluation in simulation pr | — | **新建** | [paper-kadian-sim2real-predictivity](../../wiki/entities/paper-kadian-sim2real-predictivity.md) |
| 39 | 视觉 Sim2Real | SplatSim: zero-shot sim2real transfer of RGB manipulati | [2409.10161](https://arxiv.org/abs/2409.10161) | **新建** | [paper-splatsim-gaussian-splatting-sim2real](../../wiki/entities/paper-splatsim-gaussian-splatting-sim2real.md) |
| 40 | 视觉 Sim2Real | GaussGym: an open-source real-to-sim framework for lear | [2510.15352](https://arxiv.org/abs/2510.15352) | **复用** | [paper-notebook-gaussgym-an-open-source-real-to-sim-framework-fo](../../wiki/entities/paper-notebook-gaussgym-an-open-source-real-to-sim-framework-fo.md) |
| 41 | 可微仿真 | Learning deployable locomotion control via differentiab | [2404.02887](https://arxiv.org/abs/2404.02887) | **新建** | [paper-schwarke-differentiable-simulation-locomotion-corl](../../wiki/entities/paper-schwarke-differentiable-simulation-locomotion-corl.md) |
| 42 | 训练成本 | Learning sim-to-real humanoid locomotion in 15 minutes | [2512.01996](https://arxiv.org/abs/2512.01996) | **复用** | [paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m](../../wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md) |
| 43 | 综述 | A survey of sim-to-real methods in RL: progress, prospe | [2502.13187](https://arxiv.org/abs/2502.13187) | **复用** | [paper-survey-sim2real-rl-foundation-models](../../wiki/entities/paper-survey-sim2real-rl-foundation-models.md) |
| 44 | 资源 | Awesome Humanoid Robot Learning | — | **新建** | [paper-awesome-humanoid-robot-learning](../../wiki/entities/paper-awesome-humanoid-robot-learning.md) |

- **44/44 独立节点**；阅读坐标：[freedof-sim2real-44-papers-technology-map](../../wiki/overview/freedof-sim2real-44-papers-technology-map.md)

