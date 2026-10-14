---
type: overview
tags: [frontier-papers, manipulation, vla, world-action-model]
status: complete
updated: 2026-10-12
related:
  - ../tasks/manipulation.md
  - ../tasks/loco-manipulation.md
  - frontier-manipulation-2026-09-28-10-02.md
sources:
  - ../../sources/blogs/frontier_manipulation_2026_10_05_10_09.md
summary: "这份清单覆盖 VLA 与世界–动作模型（WAM）的加速与可组合化、强化学习后训练、生成式动作策略、双臂与灵巧操作、可变形物体、人形上半身操作、长程推理规划，以及触觉与跨本体数据集和硬件。共 34 条论文条目，每条各有一页可点开；与另一篇周报重复的条目共用同一页。"
---

# 【10.5-10.9前沿论文动态】Manipulation34篇

这份清单覆盖 VLA 与世界–动作模型（WAM）的加速与可组合化、强化学习后训练、生成式动作策略、双臂与灵巧操作、可变形物体、人形上半身操作、长程推理规划，以及触觉与跨本体数据集和硬件。共 34 条论文条目，每条各有一页可点开；与另一篇周报重复的条目共用同一页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 以视觉、语言条件生成机器人动作的策略 |
| WAM | World-Action Model | 联合预测环境变化与机器人动作的模型路线 |
| RL | Reinforcement Learning | 通过与环境交互优化机器人行为 |

## 为什么重要

- 把一周内的论文动态保留为可追溯来源，同时把每篇研究的细节放到各自的页面，各页核对了 arXiv 摘要/HTML 与项目页开源状态；HTML 附录等未读到的部分已在页内标注。
- 周报的“问题/创新点”只是线索；成功率等数字来自各论文自己的任务协议，本页不做横向排名。
- 多个节点的“局限”里记录了论文正文与表格数字不一致、HTML 附录未读到等核实边界，复现前请回到原文。

## 阅读路线

```mermaid
flowchart LR
  S0["VLA / 通用操作策略"] --> S1["WAM / World Model 与操作预测"] --> S2["在线/离线强化学习与后训练"] --> S3["模仿学习、视觉运动策略与生成式动作策略"] --> S4["双臂操作"] --> S5["灵巧操作"] --> S6["柔性/可变形物体操作"] --> S7["人形机器人上半身操作"] --> S8["长程任务、推理与规划"] --> S9["数据集、预训练、跨本体与硬件平台"]
```

## 独立论文详情

### VLA / 通用操作策略

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [ViDAL：视觉动力学锚定的 VLA 动作潜空间](../entities/paper-vidal-visual-dynamics-action-latent-vla.md) | [2610.08150](https://arxiv.org/abs/2610.08150) | VLA 的动作表示主要从机器人示范中学习，动作特征与环境实际发生的视觉变化之间缺少明确联系。 |
| [EgoLAP：用语言动作推理从第一视角人类数据学机器人](../entities/paper-egolap-egocentric-language-action-reasoning.md) | [2610.08726](https://arxiv.org/abs/2610.08726) | 人类第一视角视频包含丰富的操作经验，但缺少机器人关节动作标签。 |
| [VLA-ZO：VLA 的快速零阶适应](../entities/paper-vla-zo-zeroth-order-adaptation.md) | [2610.06271](https://arxiv.org/abs/2610.06271) | VLA 部署到新相机视角或新环境后，需要适应视觉分布变化。 |
| [PLaW-VLA：预测式潜在世界建模的 VLA 策略](../entities/paper-plaw-vla-predictive-latent-world-model-vla.md) | [2610.12285](https://arxiv.org/abs/2610.12285) | VLA 主要依赖当前视觉信息生成动作，对动作执行后的环境变化缺少显式预测。 |

### WAM / World Model 与操作预测

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [OpenWAM（Stanford）：可组合世界–动作模型框架](../entities/paper-openwam-composable-world-action-models.md) | [2610.07922](https://arxiv.org/abs/2610.07922) | 现有 WAM 同时改变视频生成 Backbone、动作网络、模态交互方式和推理流程，很难判断性能提升来自哪一个设计。 |
| [LAC-WM：基于潜在动作的跨本体机器人基础世界模型](../entities/paper-lac-wm-cross-embodiment-latent-action-world-model.md) | [2610.10846](https://arxiv.org/abs/2610.10846) | 不同机器人具有不同的关节结构和动作空间。 |
| [MiniWAM：用紧凑未来目标做高效世界–动作建模](../entities/paper-miniwam-compact-future-targets.md) | [2610.12194](https://arxiv.org/abs/2610.12194) | WAM 通常预测高维未来视觉特征，导致大量训练计算被用于重建与机器人控制关系不大的视觉信息。 |
| [RealtimeWAM：免训练的世界–动作模型实时推理](../entities/paper-realtimewam-training-free-wam-acceleration.md) | [2610.10079](https://arxiv.org/abs/2610.10079) | WAM 同时承担未来世界预测和机器人动作生成，视频生成过程带来的计算延迟限制了模型在真实机器人闭环控制中的应用。 |
| [Long-WAM：Scaling the Context of World-Action Models](../entities/paper-long-wam-scaling-context.md) | [2610.10528](https://arxiv.org/abs/2610.10528) | 短时间窗口难以提供长程机器人操作所需的完整历史信息。 |
| [UNITAS](../entities/paper-unitas-3d-native-wam.md) | [2610.12099](https://arxiv.org/abs/2610.12099) | 基于二维视频生成的 WAM 需要间接学习三维几何关系。 |
| [WAM-Cache](../entities/paper-wam-cache.md) | [2610.11401](https://arxiv.org/abs/2610.11401) | WAM 在连续控制过程中会反复处理相似的历史视觉和动作上下文，导致大量重复计算。 |

### 在线/离线强化学习与后训练

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [DRIVE（多样性驱动的 VLA 强化微调）](../entities/paper-drive-diversity-rl-vla.md) | [2610.09943](https://arxiv.org/abs/2610.09943) | RL Fine-Tuning 能提高 VLA 在训练任务上的成功率，但策略可能逐渐集中到少数成功动作模式。 |
| [EigenDEXplore](../entities/paper-eigendexplore.md) | [2610.07681](https://arxiv.org/abs/2610.07681) | 多指灵巧手的动作空间维度高，直接在所有关节上进行随机探索，容易产生大量无效动作，降低强化学习的样本效率。 |
| [PathTime-VLA](../entities/paper-pathtime-vla.md) | [2610.11771](https://arxiv.org/abs/2610.11771) | VLA 动作轨迹同时包含空间路径和执行时间信息。 |

### 模仿学习、视觉运动策略与生成式动作策略

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [Immiscible Diffusion Policy](../entities/paper-immiscible-diffusion-policy.md) | [2610.09369](https://arxiv.org/abs/2610.09369) | 同一机器人状态可能对应多种合理动作，例如从不同方向抓取同一物体。 |
| [ESP（Energy-Score Policy）](../entities/paper-esp-energy-score-policy.md) | [2610.07696](https://arxiv.org/abs/2610.07696) | Diffusion Policy 能生成多种动作，但通常需要多步迭代去噪。 |
| [RoboPace](../entities/paper-robopace.md) | [2610.09696](https://arxiv.org/abs/2610.09696) | Action Chunk 通常包含固定的动作时间安排，但机器人在自由空间移动和与物体接触时，对速度和执行精度的要求不同。 |

### 双臂操作

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [AE-VLA 与 ACG-Bench](../entities/paper-ae-vla-arm-wise-compositional.md) | [2610.06184](https://arxiv.org/abs/2610.06184) | 双臂机器人可能已经分别学会多个单臂技能，但当任务要求重新分配左右臂角色、改变技能执行顺序或增加同步约束时，策略往往无法直接组合已有能力。 |
| [CoRE（Collaboration-Role Experts）](../entities/paper-core-collaboration-role-experts.md) | [2610.07752](https://arxiv.org/abs/2610.07752) | 多机器人协同操作需要不同执行单元承担不同角色。 |

### 灵巧操作

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [OmniHOI](../entities/paper-omnihoi.md) | [2610.10855](https://arxiv.org/abs/2610.10855) | 人类操作视频包含大量灵巧手动作，但单目视频缺少完整三维信息，手与物体之间还存在遮挡。 |
| [OmniDex](../entities/paper-omnidex.md) | [2610.11194](https://arxiv.org/abs/2610.11194) | 灵巧手在简单场景中能够完成抓取，但在多个物体相互遮挡、接触关系复杂的杂乱场景中，抓取策略的泛化能力仍然有限。 |

### 柔性/可变形物体操作

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [Point It, Strike It](../entities/paper-point-it-strike-it.md) | [2610.09573](https://arxiv.org/abs/2610.09573) | 绳索、线缆等可变形线状物体具有复杂的动态行为。 |
| [Same Action, Different Outcome](../entities/paper-same-action-different-outcome-cloth.md) | [2610.10801](https://arxiv.org/abs/2610.10801) | 即使机器人重复执行完全相同的动作，布料也可能产生明显不同的最终形态。 |

### 人形机器人上半身操作

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [InterMimicGen：用自演进动作模仿扩展人形移动操作](../entities/paper-intermimicgen.md) | [2610.06850](https://arxiv.org/abs/2610.06850) | 人形机器人移动操作需要同时协调行走、全身姿态、手臂运动和物体交互。 |
| [HULK：重载人形机器人的全身移动操作](../entities/paper-hulk-forceful-loco-manipulation.md) | [2610.08970](https://arxiv.org/abs/2610.08970) | 人形机器人在推、拉、搬运等任务中，需要持续向环境施加较大的接触力。 |
| [Workhorse：从人类数据学习稳健全身移动操作](../entities/workhorse-humanoid-loco-manipulation.md) | [2610.09117](https://arxiv.org/abs/2610.09117) | 人类操作数据包含自然的全身协调行为，但人类与机器人在动力学、关节结构和接触约束上存在差异，直接模仿容易产生不稳定或不可执行动作。 |
| [I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning](../entities/paper-i-bfm.md) | [2610.06129](https://arxiv.org/abs/2610.06129) | 人形机器人需要适应多种外部接触和交互任务。 |

### 长程任务、推理与规划

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [PACE](../entities/paper-pace-progress-aligned-context.md) | [2610.07917](https://arxiv.org/abs/2610.07917) | 长程操作通常包含多个连续阶段。 |
| [Detect-Restore-Resume（OSS 技能衔接恢复）](../entities/paper-oss-detect-restore-resume.md) | [2610.10810](https://arxiv.org/abs/2610.10810) | 长程任务经常通过串联多个独立训练的技能完成。 |
| [SkillWeave](../entities/paper-skillweave.md) | [2610.12046](https://arxiv.org/abs/2610.12046) | 长程机器人操作往往需要多种技能，而这些技能可能来自不同来源、不同数据采集方式或不同操作任务。 |

### 数据集、预训练、跨本体与硬件平台

| 论文节点 | arXiv | 文章中的问题线索 |
|----------|-------|------------------|
| [BiGym 2.0：人形家庭操作基准](../entities/paper-bigym2-humanoid-household-benchmark.md) | [2610.07594](https://arxiv.org/abs/2610.07594) | 人形家庭操作任务涉及不同物体、复杂环境和较长执行过程。 |
| [OpenViTac](../entities/paper-openvitac.md) | [2610.10384](https://arxiv.org/abs/2610.10384) | 视觉触觉机器人策略需要同时处理图像、接触状态和动作信息，但不同研究使用的触觉传感器、仿真环境与真实机器人接口不统一，增加了算法复现和 Sim-to-Real 评测难度。 |
| [TACROSS](../entities/paper-tacross.md) | [2610.11945](https://arxiv.org/abs/2610.11945) | 灵巧操作学习需要大量真实接触数据，但不同触觉传感器在结构、分辨率和输出形式上存在差异，导致触觉数据难以统一采集与复用。 |
| [ManiUnit](../entities/paper-maniunit.md) | [2610.12089](https://arxiv.org/abs/2610.12089) | 长程操作任务需要将多个基础技能组合起来执行，但现有数据集往往更关注单个操作任务，缺少针对技能组合与长程执行能力的统一评测。 |

## 关联页面

- [任务页（一）](../tasks/manipulation.md)
- [任务页（二）](../tasks/loco-manipulation.md)
- [上一期周报](frontier-manipulation-2026-09-28-10-02.md)

## 参考来源

- [【10.5-10.9前沿论文动态】Manipulation34篇 综述条目](../../sources/blogs/frontier_manipulation_2026_10_05_10_09.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/oPAUE5EvKeOfGU-Vjsynrw)

## 推荐继续阅读

- [具身运控lab 往期周报：9.28–10.2 人形/四足](frontier-humanoid-quadruped-2026-09-28-10-02.md)
- [arXiv cs.RO 最新列表](https://arxiv.org/list/cs.RO/recent)
