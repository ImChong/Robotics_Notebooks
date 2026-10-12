---
type: overview
tags: [frontier-papers, humanoid, quadruped, locomotion]
status: complete
updated: 2026-10-12
related:
  - ../tasks/locomotion.md
  - ../tasks/loco-manipulation.md
  - frontier-humanoid-quadruped-2026-09-28-10-02.md
sources:
  - ../../sources/blogs/frontier_humanoid_quadruped_2026_10_05_10_09.md
summary: "这份清单覆盖人形动态移动操作与柔顺控制、遥操作与人体数据重定向、人形世界–动作模型与触觉导航，以及四足载荷运输、空地协同、能耗步态、月面与屋顶地形和轮足竞速。共 44 条论文条目，每条对应一个独立详情节点；与另一篇周报重复的条目共用同一节点。"
---

# 【10.5-10.9前沿论文动态】人形/四足44篇

这份清单覆盖人形动态移动操作与柔顺控制、遥操作与人体数据重定向、人形世界–动作模型与触觉导航，以及四足载荷运输、空地协同、能耗步态、月面与屋顶地形和轮足竞速。共 44 条论文条目，每条对应一个独立详情节点；与另一篇周报重复的条目共用同一节点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 以视觉、语言条件生成机器人动作的策略 |
| WAM | World-Action Model | 联合预测环境变化与机器人动作的模型路线 |
| RL | Reinforcement Learning | 通过与环境交互优化机器人行为 |

## 为什么重要

- 把一周内的论文动态保留为可追溯来源，同时把每篇研究的细节拆到稳定的独立节点，各节点核对了 arXiv 摘要/HTML 与项目页开源状态；HTML 附录等未读到的部分已在节点内标注。
- 周报的“问题/创新点”只是线索；成功率等数字来自各论文自己的任务协议，本页不做横向排名。
- 多个节点的“局限”里记录了论文正文与表格数字不一致、HTML 附录未读到等核实边界，复现前请回到原文。

## 阅读路线

```mermaid
flowchart LR
  S0["人形机器人"] --> S1["四足机器人"] --> S2["轮足与通用腿式机器人"]
```

## 独立论文详情

### 人形机器人

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [Dataset-Free Compliant Humanoid Loco-Manipulation with Dynamic Online Posture（OCLO）](../entities/paper-oclo-online-posture-compliant-loco-manipulation.md) | [2610.05678](https://arxiv.org/abs/2610.05678) | 仅给定双手末端目标时，机器人既缺少全身参考动作，也难以兼顾可达性、平衡和接触柔顺性。 |
| [BRACE：让全身跟踪同时适应地形与交互力](../entities/paper-brace-force-terrain-whole-body-tracking.md) | [2610.07052](https://arxiv.org/abs/2610.07052) | 平地动作参考不能直接适用于斜坡、台阶以及双手持续施力场景。 |
| [欠驱动双足分层 RL 无碰撞运动](../entities/paper-hrl-underactuated-biped-collision-free.md) | [2610.05855](https://arxiv.org/abs/2610.05855) | 欠驱动双足机器人需要在保持动态平衡的同时，根据局部障碍观测完成目标导航。 |
| [I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning](../entities/paper-i-bfm.md) | [2610.06129](https://arxiv.org/abs/2610.06129) | 现有通用人形行为模型主要生成无接触动作，物体交互仍依赖任务专用参考轨迹或单独训练。 |
| [TRABot：说、演、动](../entities/paper-trabot-talk-render-act.md) | [2610.06153](https://arxiv.org/abs/2610.06153) | 对话机器人语音、面部动画和身体手势通常独立生成，容易出现语义与时间不同步。 |
| [R2RI：机器人间交互的多视角事件与 RGB 数据集](../entities/paper-r2ri-robot-interaction-dataset.md) | [2610.07117](https://arxiv.org/abs/2610.07117) | 机器人间交互研究缺少同时包含第一视角、第三视角及事件数据的大规模数据集。 |
| [InterMimicGen：用自演进动作模仿扩展人形移动操作](../entities/paper-intermimicgen.md) | [2610.06850](https://arxiv.org/abs/2610.06850) | 人形移动操作数据分散在不同人体—物体数据源中，动作质量和机器人可执行性差异很大。 |
| [高程图看不到什么：语义感知运动与执行感知导航](../entities/paper-semantic-aware-locomotion-execution-aware-nav.md) | [2610.07396](https://arxiv.org/abs/2610.07396) | 高程图无法表示湿滑、脆弱、危险区域等语义风险，导航器也通常忽略实际运动控制器对指令的执行误差。 |
| [BiGym 2.0：人形家庭操作基准](../entities/paper-bigym2-humanoid-household-benchmark.md) | [2610.07594](https://arxiv.org/abs/2610.07594) | 缺少覆盖人形全身家庭操作、原生人类示范以及多类学习方法的统一基准。 |
| [Beyond Retargeting：学习式原子动作基元的低延迟人形全身遥操作](../entities/paper-atomic-motion-primitives-teleoperation.md) | [2610.07891](https://arxiv.org/abs/2610.07891) | 直接人体动作重定向容易产生机器人不可执行姿态，且复杂低层跟踪模型会增加遥操作延迟。 |
| [AutodidactWAM：从生成视频到机器人动作的跨模态自蒸馏](../entities/paper-autodidactwam.md) | [2610.08119](https://arxiv.org/abs/2610.08119) | 生成视频能够扩充任务和场景，但缺少与视频对应的可执行机器人动作标签。 |
| [iGPC：面向物体感知人形交互的生成式动作先验](../entities/paper-igpc-object-aware-humanoid-interaction.md) | [2610.08120](https://arxiv.org/abs/2610.08120) | 传统人体动作先验缺少物体和接触条件，无法可靠支持人形机器人在真实场景中的交互。 |
| [Humanoid Horizon：用并行训练、动态起始与奖励门控扩展全身移动操作时域](../entities/paper-humanoid-horizon-long-horizon-loco-manipulation.md) | [2610.08320](https://arxiv.org/abs/2610.08320) | 多物体连续搬运中，强化学习容易偏向前期简单阶段，并在训练后期任务时遗忘先前行为。 |
| [From Legs to Wheels：面向轮式人形的形态感知人体动作重定向](../entities/paper-legs-to-wheels-mobile-base-humanoid-retargeting.md) | [2610.08381](https://arxiv.org/abs/2610.08381) | 人体下肢动作无法直接映射到没有腿部关节的轮式人形机器人。 |
| [HULK：重载人形机器人的全身移动操作](../entities/paper-hulk-forceful-loco-manipulation.md) | [2610.08970](https://arxiv.org/abs/2610.08970) | 搬运大型重物会改变质心并持续向上肢施加载荷，使平衡与动作跟踪相互冲突。 |
| [MimicX（策略在环的视频驱动人形动作跟踪监督精炼）](../entities/paper-mimicx.md) | [2610.09055](https://arxiv.org/abs/2610.09055) | 从视频重建的动作视觉上合理，但可能包含持续不可执行的转场、碰撞或局部姿态。 |
| [Workhorse：从人类数据学习稳健全身移动操作](../entities/workhorse-humanoid-loco-manipulation.md) | [2610.09117](https://arxiv.org/abs/2610.09117) | 人类示范不能直接解决人形机器人平衡、接触误差和任务失败后的自主恢复。 |
| [ResGAC：全身人形控制中的精确 SE(3) 末端执行器跟踪](../entities/paper-resgac-precise-se3-end-effector-tracking.md) | [2610.09479](https://arxiv.org/abs/2610.09479) | 全身运动策略通常难以同时实现精确末端位置、姿态跟踪和接触柔顺性。 |
| [RoboSteer：行为基础模型可操控性基准](../entities/paper-robosteer-behavior-steerability-benchmark.md) | [2610.10198](https://arxiv.org/abs/2610.10198) | 现有评测只关注人形行为能否生成，无法判断模型是否准确执行速度、方向、顺序和身体部位等复合要求。 |
| [HuMBLE：人类动作驱动的具身 locomotion](../entities/paper-humble-human-motion-driven-behavior.md) | [2610.10489](https://arxiv.org/abs/2610.10489) | 命令跟踪策略步态机械，而严格模仿人体动作的策略又难以响应数据分布外的速度与方向指令。 |
| [TAPNAV：通过触觉主动感知实现人形机器人导航](../entities/paper-tapnav-humanoid-tactile-navigation.md) | [2610.10748](https://arxiv.org/abs/2610.10748) | 在黑暗、烟雾或摄像头失效时，人形机器人无法仅靠视觉确认障碍与通道结构。 |
| [Being-M0.7: A Latent World-Action Model for Humanoid Robots（人形机器人潜空间世界–动作模型）](../entities/paper-being-m07-humanoid-latent-wam.md) | [2610.11283](https://arxiv.org/abs/2610.11283) | 人形移动操作数据昂贵，而大量人类数据往往只有视频或动作之一，不能直接生成机器人控制命令。 |
| [DAMP：去噪信念学习 + 对抗动作先验的人形运动控制](../entities/paper-damp-humanoid-denoised-belief-amp.md) | [2610.11505](https://arxiv.org/abs/2610.11505) | 没有外部地形感知时，人形机器人难以在复杂地形上同时保持稳定、目标一致和自然动作。 |
| [HWAM：联合状态–动作生成的人形世界动作模型](../entities/paper-hwam-humanoid-world-action-model.md) | [2610.12026](https://arxiv.org/abs/2610.12026) | 高层策略输出的参考动作经过平衡控制、动力学和接触后会产生执行偏差，普通世界动作模型没有显式建模这一差距。 |
| [VioLA：从人类数据学习人形通用控制](../entities/paper-viola-human-data-control.md) | [2610.12435](https://arxiv.org/abs/2610.12435) | 人形关节动作维度高且强耦合，机器人示范数量又远少于人类视频和动作数据。 |
| [DASH：面向串联机械臂与人形机器人的达芬奇器械适配器](../entities/paper-dash-da-vinci-adapter.md) | [2610.05792](https://arxiv.org/abs/2610.05792) | 达芬奇手术器械高度依赖专用平台，难以用于通用机械臂和人形机器人研究。 |
| [PhoneBot：复用智能手机的低成本开源人形平台](../entities/paper-phonebot-smartphone-humanoid.md) | [2610.08737](https://arxiv.org/abs/2610.08737) | 小型人形平台仍需要独立计算、视觉、通信和交互硬件，导致成本及复现门槛较高。 |

### 四足机器人

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [PAMORT：用多目标强化学习运输未固定的堆叠载荷](../entities/paper-pamort-stacked-payload-quadruped-morl.md) | [2610.05819](https://arxiv.org/abs/2610.05819) | 机器人自身稳定并不代表未固定箱体不会滑动或倾倒，速度跟踪与载荷稳定之间存在动态冲突。 |
| [四足机器人为微型无人机提供定位与主动跟踪](../entities/paper-quadruped-uav-localization-active-tracking.md) | [2610.06215](https://arxiv.org/abs/2610.06215) | 微型无人机无法携带完整定位传感器和计算平台。 |
| [四足机器人鲁棒非抓取式物体运输](../entities/paper-quadruped-nonprehensile-object-transport.md) | [2610.07245](https://arxiv.org/abs/2610.07245) | 背部托盘上的无约束物体对惯量误差和机器人加速度高度敏感，容易滑落。 |
| [Feeling Through the Load：载荷交互下的四足柔顺运动](../entities/paper-feeling-through-the-load-compliant-quadruped.md) | [2610.08637](https://arxiv.org/abs/2610.08637) | 人直接推动未固定载荷时，机器人需要区分载荷自然滑动与有意引导力，并安全改变运动。 |
| [腿式机器人的磁场感知控制](../entities/paper-magnet-aware-legged-control.md) | [2610.08653](https://arxiv.org/abs/2610.08653) | 强磁场会破坏传感器并对机器人铁磁部件产生随位姿变化的力和力矩，普通扰动控制器难以补偿。 |
| [四足多技能：RL+模仿联合蒸馏与对抗任务选择](../entities/paper-quadruped-skills-adversarial-task-selection.md) | [2610.10601](https://arxiv.org/abs/2610.10601) | 在一个策略中同时学习行走、挖掘、跳跃等差异显著的任务会产生样本效率低和梯度冲突。 |
| [Borrowed Eyes：四足主动观察者支持的无标记纳米无人机飞行](../entities/paper-borrowed-eyes-quadruped-nano-uav.md) | [2610.09967](https://arxiv.org/abs/2610.09967) | 无人机携带标记会增加重量和阻力，而快速运动与遮挡又容易使外部观察丢失目标。 |
| [分层强化学习节能步态适配（四足多地形）](../entities/paper-hrl-energy-efficient-gait-adaptation-quadruped.md) | [2610.10297](https://arxiv.org/abs/2610.10297) | 固定步态难以同时适应速度和地形变化，直接最小化能耗又可能损害运动稳定性。 |
| [CMP-IRRT*：感知辅助的四足高度自适应规划器](../entities/paper-cmp-irrt-star-height-adaptive-planner.md) | [2610.10470](https://arxiv.org/abs/2610.10470) | 二维规划器通常把所有障碍都视为不可穿越，无法利用四足机器人跨越低矮物体的能力。 |
| [高阶形态先验：Go1 执行器退化下的 Hodge 消息传递策略](../entities/paper-hodge-morphology-priors-quadruped-degradation.md) | [2610.10934](https://arxiv.org/abs/2610.10934) | 某些关节驱动力下降后，其余关节必须形成跨肢体补偿，普通图策略只建模成对连接。 |
| [RAGNAROK：四足的雷达–视觉–运动学–惯性 SLAM](../entities/paper-ragnarok-radar-visual-kinematic-inertial-slam.md) | [2610.11531](https://arxiv.org/abs/2610.11531) | 四足机器人在打滑、冲击、低光、强眩光和重复纹理环境中，单一视觉或本体里程计容易失效。 |
| [Walking on Roofs：商用四足能在屋顶上干活吗](../entities/paper-walking-on-roofs-quadruped-go2.md) | [2610.12272](https://arxiv.org/abs/2610.12272) | 商业四足机器人及标准球形足端是否适合倾斜屋顶尚缺少系统验证。 |
| [Toward Lunar Legged Robots：LUNA 月面模拟场的部署教训](../entities/paper-toward-lunar-legged-robots-luna-field.md) | [2610.12276](https://arxiv.org/abs/2610.12276) | 月壤沉陷、打滑、扬尘和极端照明对四足机器人运动、感知和任务运行的影响缺少长时实地数据。 |

### 轮足与通用腿式机器人

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [RACER：用残差闭环模型让 MPPI 驾驭轮足高速过弯](../entities/paper-racer-residual-adaptive-wheeled-quadruped-racing.md) | [2610.07409](https://arxiv.org/abs/2610.07409) | 高速转弯时，简单运动学模型无法预测RL速度跟踪器受轮胎打滑、延迟和执行器饱和影响后的真实闭环行为。 |
| [2GO：几何感知的模型式生成优化做受约束腿式规划](../entities/paper-2go-geometry-aware-generative-locomotion-planning.md) | [2610.07772](https://arxiv.org/abs/2610.07772) | 腿式机器人在强接触、碰撞及运动学约束下，传统优化容易陷入局部解，纯生成模型又难以严格满足约束。 |
| [LLA-MPPI：用 GPU 并行仿真做腿式全身控制的在线自适应](../entities/paper-lla-mppi-legged-adaptive-wbc.md) | [2610.10465](https://arxiv.org/abs/2610.10465) | 载荷、摩擦、坡度和执行器故障会在线改变接触动力学，而固定模型控制器和离线训练策略难以覆盖未知变化。 |
| [ClimbLab：腿式攀爬机器人的 MATLAB 仿真平台](../entities/paper-climblab-matlab-legged-climbing-sim.md) | [2610.09315](https://arxiv.org/abs/2610.09315) | 腿式攀爬研究缺少便于教学、动力学分析和控制原型验证的轻量开源仿真工具。 |

## 关联页面

- [任务页（一）](../tasks/locomotion.md)
- [任务页（二）](../tasks/loco-manipulation.md)
- [上一期周报](frontier-humanoid-quadruped-2026-09-28-10-02.md)

## 参考来源

- [【10.5-10.9前沿论文动态】人形/四足44篇 综述条目](../../sources/blogs/frontier_humanoid_quadruped_2026_10_05_10_09.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/HPqfC2jkqoL4zPn9c9pGEw)

## 推荐继续阅读

- [具身运控lab 往期周报：9.28–10.2 人形/四足](frontier-humanoid-quadruped-2026-09-28-10-02.md)
- [arXiv cs.RO 最新列表](https://arxiv.org/list/cs.RO/recent)
