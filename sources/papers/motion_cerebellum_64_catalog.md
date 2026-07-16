# 运动小脑 64 篇论文 source 索引

> 来源归档（catalog）

- **微信公众号导读：** [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- **wiki 技术地图：** [humanoid-motion-cerebellum-technology-map.md](../../wiki/overview/humanoid-motion-cerebellum-technology-map.md)
- **原始链接：** <https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA>
- **入库日期：** 2026-06-18
- **一句话说明：** 具身智能研究室「动作小脑」长文所列 **64 篇** 人形运控论文策展索引；**复用** 既有 `paper-hrl-stack-*` / `paper-loco-manip-*` / `paper-amp-survey-*` 等实体，仅为 **15 篇** 尚无索引的工作新建 `paper-motion-cerebellum-*` 节点。

## A. 走路底座（10）

| # | 工作 | 角色 | Wiki 实体 |
|---|------|------|-----------|
| 01 | GuideWalk | 底座：把导航接口接到地形自适应步态 | [GuideWalk](../../wiki/entities/paper-motion-cerebellum-guidewalk.md) |
| 02 | T-GMP | 底座：用地形条件运动先验改善自然步态 | [T-GMP](../../wiki/entities/paper-motion-cerebellum-t-gmp.md) |
| 03 | PerceptiveBFM | 底座：让行为基座模型看见地形 | [PerceptiveBFM](../../wiki/entities/paper-perceptive-bfm.md) |
| 04 | MARCH | 底座：模型辅助稀疏落脚控制 | [MARCH](../../wiki/entities/paper-motion-cerebellum-march.md) |
| 05 | AMS | 底座：敏捷性和稳定性的多数据融合 | [AMS](../../wiki/methods/ams.md) |
| 06 | OmniXtreme | 底座：高动态技能把稳定边界往外推 | [OmniXtreme](../../wiki/entities/paper-hrl-stack-16-omnixtreme.md) |
| 07 | PHP | 底座：跑酷技能链和运动匹配 | [PHP](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md) |
| 08 | Deep Whole-body Parkour | 底座：手脚躯干一起参与跑酷 | [Deep Whole-body Parkour](../../wiki/entities/paper-deep-whole-body-parkour.md) |
| 09 | TAGA | 底座：主动凝视进入敏捷运动闭环 | [TAGA](../../wiki/entities/paper-motion-cerebellum-taga.md) |
| 10 | SSR | 底座：第一视角视觉驱动开放世界穿越 | [SSR](../../wiki/entities/paper-ssr-humanoid-open-world-traversal.md) |

## B. 动作模仿源流（5）

| # | 工作 | 角色 | Wiki 实体 |
|---|------|------|-----------|
| 11 | DeepMimic | 源流：参考动作 + 强化学习 | [DeepMimic](../../wiki/methods/deepmimic.md) |
| 12 | AMP | 源流：用对抗运动先验让动作像人 | [AMP](../../wiki/entities/paper-amp-survey-01-amp.md) |
| 13 | SMP | 源流：用 score-matching 学可复用运动先验 | [SMP](../../wiki/methods/smp.md) |
| 14 | PHC | 跟踪：大规模动作模仿与失败恢复 | [PHC](../../wiki/entities/phc.md) |
| 15 | MaskedMimic | 跟踪：用遮蔽补全把稀疏目标变成全身动作 | [MaskedMimic](../../wiki/entities/paper-bfm-17-maskedmimic.md) |

## C. 数据入口（9）

| # | 工作 | 角色 | Wiki 实体 |
|---|------|------|-----------|
| 16 | GVHMR | 数据入口：视频动作恢复到重力对齐世界坐标 | [GVHMR](../../wiki/entities/gvhmr.md) |
| 17 | TRAM | 数据入口：野外视频到全局人体轨迹 | [TRAM](../../wiki/entities/paper-motion-cerebellum-tram.md) |
| 18 | GMR | 重定向：人类动作变成可跟踪机器人参考 | [GMR](../../wiki/entities/paper-hrl-stack-01-retargeting_matters.md) |
| 19 | NMR | 重定向：神经重定向与物理修正数据 | [NMR](../../wiki/entities/paper-hrl-stack-02-make_tracking_easy.md) |
| 20 | OmniRetarget | 重定向：交互关系保持的数据生成 | [OmniRetarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md) |
| 21 | HumanX | 数据入口：从人类视频学敏捷交互技能 | [HumanX](../../wiki/entities/paper-hrl-stack-05-humanx.md) |
| 22 | HDMI | 数据入口：人类视频到交互式全身控制 | [HDMI](../../wiki/entities/paper-hrl-stack-06-hdmi.md) |
| 23 | SUGAR | 数据入口：视频驱动的泛化移动操作数据 | [SUGAR](../../wiki/entities/paper-loco-manip-161-076-sugar.md) |
| 24 | GenMimic | 数据入口：生成视频到物理可执行轨迹 | [GenMimic](../../wiki/entities/paper-hrl-stack-04-from_generated_human_videos_to_physi.md) |

## D. 全身跟踪基座（13）

| # | 工作 | 角色 | Wiki 实体 |
|---|------|------|-----------|
| 25 | OmniTrack | 跟踪策略：物理一致参考让 tracking 更稳 | [OmniTrack](../../wiki/entities/paper-hrl-stack-12-omnitrack.md) |
| 26 | BeyondMimic | 跟踪策略：从动作跟踪到多功能人形控制 | [BeyondMimic](../../wiki/methods/beyondmimic.md) |
| 27 | SONIC | 跟踪策略：把 motion tracking 规模化成控制基座 | [SONIC](../../wiki/methods/sonic-motion-tracking.md) |
| 28 | HoloMotion-1 | 跟踪策略：视频动作也进入运动基座训练 | [HoloMotion-1](../../wiki/entities/holomotion.md) |
| 29 | HumanoidGPT | 跟踪策略：海量动作帧和 Transformer 化 | [HumanoidGPT](../../wiki/entities/paper-humanoid-gpt.md) |
| 30 | LIMMT | 跟踪策略：数据质量筛选比盲目堆量更重要 | [LIMMT](../../wiki/methods/limmt-gqs-motion-curation.md) |
| 31 | M3imic | 跟踪策略：多模态动作条件下的统一模仿 | [M3imic](../../wiki/entities/paper-loco-manip-06-m3imic.md) |
| 32 | RGMT | 鲁棒性：泛化和扰动下的动作跟踪 | [RGMT](../../wiki/entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md) |
| 33 | Any2Track | 鲁棒性：任意动作、任意扰动继续跟踪 | [Any2Track](../../wiki/methods/any2track.md) |
| 34 | Stubborn | 恢复：把跟踪和跌倒恢复放进统一 RL | [Stubborn](../../wiki/entities/paper-motion-cerebellum-stubborn.md) |
| 35 | ConstrainedMimic | 安全：给 motion tracking 加实时约束层 | [ConstrainedMimic](../../wiki/entities/paper-motion-cerebellum-constrainedmimic.md) |
| 36 | SafeWBC | 安全：控制屏障函数接到全身控制后面 | [SafeWBC](../../wiki/entities/paper-motion-cerebellum-safewbc.md) |
| 37 | SafeFall | 安全：失败不可避免时降低损伤 | [SafeFall](../../wiki/entities/paper-hrl-stack-41-safefall.md) |

## E. 可提示控制（4）

| # | 工作 | 角色 | Wiki 实体 |
|---|------|------|-----------|
| 38 | BFM-Zero | 可提示小脑：目标、奖励、轨迹 prompt 调用身体 | [BFM-Zero](../../wiki/entities/paper-bfm-zero.md) |
| 39 | MuGen | 可提示小脑：多技能生成式运动控制 | [MuGen](../../wiki/entities/paper-motion-cerebellum-mugen.md) |
| 40 | OMG | 可提示小脑：多模态提示到人形全身动作 | [OMG](../../wiki/entities/paper-omg-omni-modal-humanoid-control.md) |
| 41 | MotionWAM | 可提示小脑：世界动作模型把视觉和动作 latent 接起来 | [MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md) |

## F. 跨本体与遥操作（5）

| # | 工作 | 角色 | Wiki 实体 |
|---|------|------|-----------|
| 42 | Any2Any | 跨本体：把预训练 whole-body tracker 迁到新身体 | [Any2Any](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md) |
| 43 | TWIST | 遥操作：全身遥操作也是动作小脑的数据入口 | [TWIST](../../wiki/entities/paper-twist.md) |
| 44 | TWIST2 | 遥操作：可扩展、可携带的全身数据采集 | [TWIST2](../../wiki/entities/paper-twist2.md) |
| 45 | X-OP | 遥操作：MPC 重定向的跨本体全身遥操作 | [X-OP](../../wiki/entities/paper-loco-manip-08-x-op.md) |
| 46 | CLOT | 遥操作：闭环全局运动跟踪用于全身遥操作 | [CLOT](../../wiki/entities/paper-amp-survey-16-clot.md) |

## G. Loco-Manip 接口（5）

| # | 工作 | 角色 | Wiki 实体 |
|---|------|------|-----------|
| 47 | CEER | 接口：EE-root 命令连接高层和全身控制 | [CEER](../../wiki/entities/paper-motion-cerebellum-ceer.md) |
| 48 | HANDOFF | 接口：任务空间命令 + 多教师蒸馏 | [HANDOFF](../../wiki/entities/paper-motion-cerebellum-handoff.md) |
| 49 | MPC-RL | 接口：MPC 引导 RL，把模型结构写进训练 | [MPC-RL](../../wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) |
| 50 | VAIC | 接口：视觉目标到解耦全身物体交互控制 | [VAIC](../../wiki/entities/paper-loco-manip-05-vaic.md) |
| 51 | 主动空间大脑与泛化动作小脑 | 接口：上层规划与泛化动作小脑分工 | [主动空间大脑与泛化动作小脑](../../wiki/entities/paper-motion-cerebellum-active-spatial-brain-generalized-cerebellum.md) |

## H. 真实任务（8）

| # | 工作 | 角色 | Wiki 实体 |
|---|------|------|-----------|
| 52 | DoorMan | 任务：开门把视觉、接触、移动和平衡全照出来 | [DoorMan](../../wiki/entities/paper-doorman-opening-sim2real-door.md) |
| 53 | HOIST | 任务：悬挂负载操作考验后果建模 | [HOIST](../../wiki/entities/paper-motion-cerebellum-hoist.md) |
| 54 | SplitAdapter | 任务：负载变化下的分解式适配 | [SplitAdapter](../../wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md) |
| 55 | HALO | 任务：重载技能里的可微仿真和物理对齐 | [HALO](../../wiki/entities/paper-hrl-stack-39-closing_sim_to_real_gap_for_heavy_lo.md) |
| 56 | HumanoidMimicGen | 任务数据：全身规划驱动移动操作数据生成 | [HumanoidMimicGen](../../wiki/entities/paper-humanoidmimicgen.md) |
| 57 | GRAIL | 任务数据：3D 资产和视频先验生成 Loco-Manip 数据 | [GRAIL](../../wiki/entities/paper-grail.md) |
| 58 | OASIS | 任务数据：仿真数据驱动真实 Loco-Manip 部署 | [OASIS](../../wiki/entities/paper-loco-manip-04-oasis.md) |
| 59 | LadderMan | 任务：爬梯和梯上操作暴露手脚协同难题 | [LadderMan](../../wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md) |

## I. 柔顺与接触（5）

| # | 工作 | 角色 | Wiki 实体 |
|---|------|------|-----------|
| 60 | SoftMimic | 接触：柔顺全身控制也在成为 tracking 条件 | [SoftMimic](../../wiki/entities/paper-notebook-softmimic-learning-compliant-whole-body-control.md) |
| 61 | CHIP | 接触：通过 hindsight perturbation 学自适应柔顺 | [CHIP](../../wiki/entities/paper-hrl-stack-36-chip.md) |
| 62 | GentleHumanoid | 接触：人和物体接触里的上半身分寸感 | [GentleHumanoid](../../wiki/methods/gentlehumanoid-motion-tracking.md) |
| 63 | Thor | 接触：强接触环境里的全身反应 | [Thor](../../wiki/entities/paper-hrl-stack-42-thor.md) |
| 64 | WT-UMI | 接触数据：触觉和力监督进入数据链路 | [WT-UMI](../../wiki/entities/paper-loco-manip-07-wt-umi.md) |
