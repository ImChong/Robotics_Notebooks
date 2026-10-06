---
type: overview
tags: [frontier-papers, humanoid, quadruped, locomotion]
status: complete
updated: 2026-10-06
related:
  - ../tasks/locomotion.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/blogs/frontier_humanoid_quadruped_2026_09_28_10_02.md
summary: "这份人形/四足清单覆盖身体—手部预训练、人体示范重定向、动作先验、地形感知、全身安全、工具使用、多机器人协作，以及四足探索与颗粒地形运动。文章标题写「49篇」，正文实际列出 53 条 arXiv 论文；下表逐条保留正文条目，并按 arXiv 编号去重。"
---

# 【9.28–10.2 前沿论文动态】人形/四足49篇

这份人形/四足清单覆盖身体—手部预训练、人体示范重定向、动作先验、地形感知、全身安全、工具使用、多机器人协作，以及四足探索与颗粒地形运动。文章标题写「49篇」，正文实际列出 53 条 arXiv 论文；下表逐条保留正文条目，并按 arXiv 编号去重。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 以视觉、语言条件生成机器人动作的策略 |
| RL | Reinforcement Learning | 通过与环境交互优化机器人行为 |
| WAM | World-Action Model | 联合预测环境变化与机器人动作的模型路线 |

## 为什么重要

- 把一周内的论文动态保留为可追溯来源，同时将每篇研究的细节拆到稳定的独立节点。
- 通过问题、方法与论文原文入口连接，不把综述里不同任务的成功率直接横向排序。
- 未附项目页/仓库的条目只记录论文与摘要；需要复现时再逐一核查官方项目页和开源状态。

## 阅读路线

```mermaid
flowchart LR
  A["人体数据与动作先验"] --> B["全身控制与地形感知"] --> C["安全、协作与长程任务"]
```

## 独立论文详情

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [WB-WAM：面向人形移动操作的异构身体—手部联合预训练](../entities/paper-wb-wam.md) | [2609.34199](https://arxiv.org/abs/2609.34199) | 人体视频、动作捕捉与机器人数据对身体和手部的动作表示不统一 |
| [GAE：面向实时人形遥操作的通用动作专家](../entities/paper-gae-general-action-expert.md) | [2609.34233](https://arxiv.org/abs/2609.34233) | 人体动作含噪且与机器人动力学不匹配，通信和执行延迟还会破坏人机同步 |
| [基于步间预测的模型引导安全双足运动强化学习](../entities/paper-model-informed-safe-reinforcement-learning.md) | [2609.34486](https://arxiv.org/abs/2609.34486) | 纯强化学习步行策略缺少可解释的安全约束，受扰时可能给出危险落脚点 |
| [HOI-Retarget：面向人—物交互的接触中心动作重定向](../entities/paper-hoi-retarget.md) | [2609.34674](https://arxiv.org/abs/2609.34674) | 只匹配人体和机器人的关节或末端位置，不能保证手部接触物体上功能相同的位置 |
| [DexWeave：从人体示范学习灵巧人形移动操作](../entities/paper-dexweave-humanoid-loco-manipulation.md) | [2609.34724](https://arxiv.org/abs/2609.34724) | 分别处理身体、手部和物体轨迹会破坏三者之间的协调与接触关系 |
| [CoHuB：多个人形机器人协作仿真基准](../entities/paper-cohub.md) | [2609.34782](https://arxiv.org/abs/2609.34782) | 现有人形基准主要评估单机器人能力，缺少第一视角条件下的多机器人协作任务 |
| [受被动动态行走启发的节能人形运动动力学引导](../entities/paper-passive-dynamic-walking-inspired-dynamics.md) | [2609.35935](https://arxiv.org/abs/2609.35935) | 仅惩罚力矩和功率并不能直接帮助策略发现具有经济性的步间动力学协调 |
| [Uni-VLaT：人形移动操作VLA策略的全身触觉适配](../entities/paper-uni-vlat.md) | [2609.35450](https://arxiv.org/abs/2609.35450) | 纯视觉VLA难以判断遮挡后的接触状态、物体滑移及身体与环境的接触 |
| [基于离散视觉—语言—动作模型的人形移动操作](../entities/paper-holo-m.md) | [2609.35709](https://arxiv.org/abs/2609.35709) | 人形机器人腿、躯干、双臂和手部动作异构，难以使用统一的VLA动作表示 |
| [KPI：可提示的人形机器人物理交互控制内核](../entities/paper-kpi.md) | [2609.36151](https://arxiv.org/abs/2609.36151) | 开门、书写和挤压搬运等任务对轨迹、力和柔顺性的要求不同 |
| [EquivDP3：面向数据高效人形移动操作的SIM(3)不变点云编码器](../entities/paper-equivdp3.md) | [2609.36575](https://arxiv.org/abs/2609.36575) | 普通点云策略不能自然泛化到物体的旋转、平移和尺度变化，少量示范下容易过拟合 |
| [OTRetarget：基于最优传输的机器人与物体运动联合重定向](../entities/paper-otretarget.md) | [2609.36602](https://arxiv.org/abs/2609.36602) | 固定物体轨迹后只调整机器人动作，可能造成手部脱离物体或足部接触失真 |
| [跟踪并完成：从单段失败人体视频学习人形技能](../entities/paper-track-and-complete.md) | [2609.36924](https://arxiv.org/abs/2609.36924) | 视频模仿通常要求成功示范，但失败视频在失败前仍包含有用动作并隐含任务目标 |
| [基于谱技能的表达性与可组合动作表征学习](../entities/paper-spectral-skills-motion-representation.md) | [2609.37677](https://arxiv.org/abs/2609.37677) | 高层规划器需要容易预测、准确执行且可组合的低维动作接口 |
| [EgoHumanoid-V2：移动操作协调全身技能的人到人形迁移](../entities/paper-egohumanoid-v2.md) | [2609.37181](https://arxiv.org/abs/2609.37181) | 第一视角人体示范中的视觉、身体动作和接触关系与机器人本体不一致 |
| [EgoAlign：弥合长距离移动操作中的人与人形机器人差距](../entities/paper-egoalign.md) | [2609.38046](https://arxiv.org/abs/2609.38046) | 长距离人体示范缺少机器人状态，身体尺度和控制响应差异会导致示范不可执行 |
| [CrossBFM：跨人形机器人本体蒸馏共享潜在行为空间](../entities/paper-crossbfm-shared-latent-behavior.md) | [2609.38087](https://arxiv.org/abs/2609.38087) | 不同机器人独立训练得到的行为latent不对齐，无法跨本体复用 |
| [反事实视频生成实现可扩展人形移动操作学习](../entities/paper-prism-real2sim2real.md) | [2609.38172](https://arxiv.org/abs/2609.38172) | 少量真实视频无法覆盖不同物体外观、尺寸和摆放方式 |
| [GestAdapt：工作空间条件化的人形伴随语音手势生成](../entities/paper-gestadapt.md) | [2609.38400](https://arxiv.org/abs/2609.38400) | 靠近墙壁或物体时，普通语音手势可能超出机器人可用空间 |
| [面向腿式机器人的稠密时间动作重定向](../entities/paper-dense-temporal-motion-retargeting.md) | [2609.38617](https://arxiv.org/abs/2609.38617) | 跳跃等动作的时序与控制耦合，统一缩放整段动作会破坏不需要调整的部分 |
| [CEER²：人形移动操作中的方向可调末端与根部柔顺控制](../entities/paper-ceer2-directional-compliance.md) | [2609.38709](https://arxiv.org/abs/2609.38709) | 接触任务要求机器人在部分方向柔顺、其他方向保持精度，同时控制躯干对外力的响 应 |
| [以运动为基础的人形足球：多方向踢球库的任务门控强化学习](../entities/paper-locomotion-grounded-humanoid-soccer.md) | [2609.38852](https://arxiv.org/abs/2609.38852) | 现有足球策略围绕单段踢球参考训练，基础运动能力弱且技能切换复杂 |
| [NEXUS：面向地形自适应遥操作的感知式全身控制](../entities/paper-nexus-terrain-adaptive-teleoperation.md) | [2609.39000](https://arxiv.org/abs/2609.39000) | 操作者在平地运动时，机器人可能位于台阶、斜坡或平台边缘，直接复制人体动作会产 生错误接触 |
| [OccluDex：自遮挡第一视角灵巧操作的分层三维视触觉表征学习](../entities/paper-occludex.md) | [2609.39017](https://arxiv.org/abs/2609.39017) | 操作手会遮挡物体表面与接触区域，导致第一视角视觉状态估计失效 |
| [RoboAssist：面向长时手术辅助的交互式人—人形规划](../entities/paper-roboassist.md) | [2609.39384](https://arxiv.org/abs/2609.39384) | 手术流程和人员请求动态变化，完整重新规划延迟高且难以持续维持安全约束 |
| [IronMind：通过相机空间第一视角预训练扩展人形灵巧操作](../entities/paper-ironmind.md) | [2609.39403](https://arxiv.org/abs/2609.39403) | 人体手部与机器人末端结构不同，普通第一视角视频还缺少可靠躯干运动 |
| [ECHO-G：具身人形机器人伴随语音动作生成](../entities/paper-echo-g-cospeech-humanoid.md) | [2609.39575](https://arxiv.org/abs/2609.39575) | 伴随语音动作需要同时匹配语音韵律、文本语义和机器人本体约束 |
| [通过第一视角全身人体数据预训练通用人形移动操作模型](../entities/paper-lambda0-egocentric-human-pretraining.md) | [2610.00438](https://arxiv.org/abs/2610.00438) | 机器人全身操作数据昂贵，普通第一视角视频缺少可直接执行的机器人动作 |
| [迈向建筑施工人形机器人：遥操作可行性研究](../entities/paper-toward-humanoid-robots-in-construction.md) | [2610.00718](https://arxiv.org/abs/2610.00718) | 完全自主施工尚不成熟，需要评估人形遥操作能否完成真实建筑任务 |
| [基于学习式稳定性模型的人形机器人反应式多点接触](../entities/paper-reactive-humanoid-multi-contact-using.md) | [2610.00823](https://arxiv.org/abs/2610.00823) | 机器人受推后仅靠双脚无法恢复时，需要快速选择墙面等环境中的手部支撑位置 |
| [MASkillBlender：通过技能混合实现多个人形机器人分散式全身协同](../entities/paper-maskillblender.md) | [2610.01102](https://arxiv.org/abs/2610.01102) | 多机器人全身协同的联合动作空间很大，逐任务设计参考动作和奖励成本高 |
| [继续、终止还是受控跌倒：面向安全人形特技的可行性策略选择](../entities/paper-continue-abort-or-fall.md) | [2610.01397](https://arxiv.org/abs/2610.01397) | 翻腾等动作失控后，单一跟踪策略无法判断应该继续、主动终止还是保护性跌倒 |
| [HumanoidToolBench：从工具选择到移动执行的人形工具使用基准](../entities/paper-humanoidtoolbench.md) | [2610.02089](https://arxiv.org/abs/2610.02089) | 现有工具使用基准多局限于桌面机械臂，缺少移动接近和全身执行 |
| [InterEvolve：人形移动操作奖励程序的测试时演化](../entities/paper-interevolve.md) | [2610.02196](https://arxiv.org/abs/2610.02196) | 预训练控制器面对新任务时，现有技能难以由固定奖励和规划程序自动组合 |
| [面向安全人形全身跟踪的过滤器感知微调](../entities/paper-filter-aware-fine-tuning-for.md) | [2610.02341](https://arxiv.org/abs/2610.02341) | 运行时安全过滤器会修改动作并改变状态分布，与原跟踪策略形成动力学、目标和信息 不匹配 |
| [超越奖励投机：分阶段人形学习流程四层代理目标偏离](../entities/paper-beyond-reward-hacking.md) | [2610.03196](https://arxiv.org/abs/2610.03196) | 人形强化学习的失败不仅来自奖励设计，还可能来自课程门控、评估指标和参考动作 |
| [KungfuAthleteBot：从视频学习具备统一鲁棒恢复能力的高动态人形动作](../entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md) | [2610.03388](https://arxiv.org/abs/2610.03388) | 视频动作存在漂浮、穿地和抖动，没有执行力信息，也不包含失败恢复示范 |
| [面向四足运动与人机交互的本体力估计](../entities/paper-proprioceptive-force-estimation-for-quadruped.md) | [2609.34222](https://arxiv.org/abs/2609.34222) | 负载力需要被策略补偿，而牵引力又可作为人类给机器人的运动指令 |
| [基于四足侦察机器人的外感知—本体感知地形自主行星探索](../entities/paper-terrain-aware-autonomous-planetary-exploration.md) | [2609.35493](https://arxiv.org/abs/2609.35493) | 几何地图无法反映月面类地形实际引起的滑移、稳定性下降和能耗变化 |
| [ChronoSRL：自监督强化学习的时间几何表征](../entities/paper-chronosrl.md) | [2609.36238](https://arxiv.org/abs/2609.36238) | 空间上相近的目标可能因障碍和机器人能力而需要很长时间到达 |
| [从机器人足—地力历史推断土壤内摩擦角](../entities/paper-inferring-soil-friction-angle-from-robot-foot-ground-force-histories.md) | [2609.36582](https://arxiv.org/abs/2609.36582) | 四足机器人在沙土环境中需要估计土壤强度，但专用测量设备会增加负担 |
| [面向鲁棒腿式运动的预测式安全课程学习](../entities/paper-predictive-safety-curricula-for-robust.md) | [2609.37070](https://arxiv.org/abs/2609.37070) | 平均性能优秀的运动策略仍可能存在低概率碰撞，普通课程无法主动增加稀有危险经 验 |
| [基于经验驱动持续学习的四足地形可通行性预测](../entities/paper-experience-driven-continual-learning-of.md) | [2609.39755](https://arxiv.org/abs/2609.39755) | 地形外观无法直接判断滑移、足端冲击和能耗，持续更新模型又容易遗忘旧地形 |
| [本体不匹配条件下基于经验可行性感知的生成式对抗观察模仿](../entities/paper-experience-based-feasibility-aware-generative.md) | [2610.01171](https://arxiv.org/abs/2610.01171) | 人体状态轨迹可能超出四足机器人的动力学能力，直接模仿会让不可行动作误导策略 |
| [PROMO：偏好条件化的四足机器人多目标强化学习](../entities/paper-promo.md) | [2610.01260](https://arxiv.org/abs/2610.01260) | 速度跟踪、稳定性和能耗的权衡在训练后被固定，部署时无法切换 |
| [ReCo：采用策略感知MPC的响应一致腿式移动操作](../entities/paper-reco.md) | [2610.01612](https://arxiv.org/abs/2610.01612) | RL运动策略对速度指令的响应随步态相位、接触和负载变化，MPC难以准确预测 |
| [面向强化学习的双向Voronoi偏置探索课程](../entities/paper-bidirectional-voronoi-biased-exploration-curriculum.md) | [2610.03395](https://arxiv.org/abs/2610.03395) | 四足爬箱等长时稀疏奖励任务很难从初始状态探索到目标 |
| [CrowdOcc：面向拥挤室内环境四足机器人的单目语义场景补全](../entities/paper-crowdocc.md) | [2610.03031](https://arxiv.org/abs/2610.03031) | 人群遮挡会破坏室内静态几何重建，并导致人体占据位置不完整或错误 |
| [TACET：面向四足机器人的情境适配声学—社会导航](../entities/paper-tacet.md) | [2610.03828](https://arxiv.org/abs/2610.03828) | 医院和办公室中的四足机器人不仅要避让人员，还要根据场景控制行走噪声 |
| [环行世界：270克连续旋转四足机器人的统一学习式运动](../entities/paper-around-the-world.md) | [2610.02728](https://arxiv.org/abs/2610.02728) | 亚千克四足平台上的闭环学习式运动仍较少，连续旋转腿还具有普通关节腿没有的姿态 空间 |
| [LocoWM：基于世界模型引导残差适配的高精度运动控制](../entities/paper-locowm.md) | [2609.39179](https://arxiv.org/abs/2609.39179) | 普通残差控制只能在误差已经出现后进行修正，无法提前补偿未来偏差 |
| [具备游泳与步行能力的多关节仿生水下机器人多功能运动控制](../entities/paper-multifunctional-locomotion-control-of-multi.md) | [2609.37086](https://arxiv.org/abs/2609.37086) | 兼具鳍游和腿式步行能力的机器人通常依靠人工状态机切换 |
| [面向颗粒斜坡有效运动的地形依赖周期内腿部时序](../entities/paper-terrain-dependent-intra-cycle-leg.md) | [2610.04144](https://arxiv.org/abs/2610.04144) | 六足机器人在颗粒斜坡上容易下滑，但不同坡度所需的周期内腿部速度分配尚不明确 |

## 阅读说明

- 同一篇论文可能出现在两篇文章中；arXiv 编号相同就共用一个详情节点。
- 表格中的问题摘要来自附件综述，原始方法、实验设置与结论请以 arXiv 原文为准。
- 本次附件仅为论文综述，没有为每项工作附独立代码/数据链接；没有根据缺少链接推断其未开源。

## 关联页面

- [【9.28–10.2 前沿论文动态】人形/四足49篇 来源归档](../../sources/blogs/frontier_humanoid_quadruped_2026_09_28_10_02.md)
- [Locomotion任务页](../tasks/locomotion.md)
- [Loco-manipulation](../tasks/loco-manipulation.md)

## 参考来源

- [【9.28–10.2 前沿论文动态】人形/四足49篇 原文](../../sources/blogs/frontier_humanoid_quadruped_2026_09_28_10_02.md)
- [微信公众号原文](https://mp.weixin.qq.com/s/JTzpow1Ls1T4CCHSvh5mQA)

## 推荐继续阅读

- [arXiv](https://arxiv.org/)
