# HumanNet

> 来源归档（ingest）

- **标题：** HumanNet: Scaling Human-centric Video Learning to One Million Hours
- **类型：** paper
- **来源：** [arXiv:2605.06747](https://arxiv.org/abs/2605.06747)；[项目页](https://dagroup-pku.github.io/HumanNet/)；[代码与发布入口](https://github.com/DAGroup-PKU/HumanNet/)
- **机构：** 北京大学（DAGroup-PKU）
- **入库日期：** 2026-05-14
- **最后更新：** 2026-05-14
- **一句话说明：** 面向具身学习的 **约百万小时** 人中心互联网视频语料：一三人称并存、交互导向标注（字幕、运动描述、手/体相关信号）与可审计的采集–处理–标注管线；论文给出在固定下游 VLA 设定下，**1000 小时** 该语料中 egocentric 子集持续预训练相对 **约 100 小时** 真机机器人数据的受控对比结论。

## 核心论文摘录（MVP）

### 1) 问题与主张：具身数据规模瓶颈 vs 人中心视频

- **链接：** <https://arxiv.org/abs/2605.06747>
- **摘录要点：** 视觉–语言可依赖互联网多模态语料持续扩展，而物理交互模型仍受限于小规模、任务窄、平台绑定的机器人日志。人类活动视频在尺度与行为多样性上可补位：第一人称保留执行视点与手–物关系，第三人称补充全身与场景上下文；关键不在「堆时长」 alone，而在 **人中心过滤、时间结构、视点多样性、标注丰富度** 作为一等设计目标，把非结构化网页视频变成可预训练的基础设施。
- **对 wiki 的映射：**
  - [VLA](../../wiki/methods/vla.md) — 大规模 VLM/VLA 预训练的人类视频数据来源与「机器人专有数据是否唯一」的讨论
  - [Imitation Learning](../../wiki/methods/imitation-learning.md) — 人类演示 / 视频作为 IL 与跨本体迁移监督的外部规模来源

### 2) 语料定义与三阶段管线（采集 → 处理 → 标注）

- **链接：** <https://arxiv.org/html/2605.06747v1>（HTML 版便于对照图表与章节）
- **摘录要点：** **人中心视频** 定义为「人类活动是剪辑组织信号」且包含可学习的物理交互（操作、工具、导航、多步程序等），主动排除人类运动仅作背景的长镜头。管线三阶段：（1）**采集**—关键词发现/扩展、平台与网页检索、开源集与自控采集汇入统一池；（2）**处理**—去重归一化、内容/质量过滤、按视觉变化切场景、定长或边界清晰的 clip 化；（3）**标注**—3D 手体姿态、满足条件时的单目 SLAM、向统一人形骨架的 **运动重定向**（论文给出「重定向误差 <15 mm 且有效帧覆盖率 >60%」的 **robot-ready** 子集准则），以及 LLM 辅助生成视频字幕、运动描述与活动分类并与源元数据对齐。
- **对 wiki 的映射：**
  - [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) — 将人体运动变为跨本体训练信号的接口
  - [Embodied Data Cleaning](../../wiki/concepts/embodied-data-cleaning.md) — 大规模异构视频上的质量与分布控制思路（概念层对照，非等价实现）

### 3) 受控 VLA 后训练验证（egocentric 人视频 vs 真机小时）

- **链接：** <https://arxiv.org/abs/2605.06747>
- **摘录要点：** 在统一 **LingBot-VLA** 架构与固定下游机器人后训练语料（文中报告约 34 小时、100 任务×20 episode）下，仅改变持续预训练数据源：Qwen VLM 基线、叠加约 **100 小时** 真机数据（文中称 Magic Cobot / CoBot 数据）、叠加自 HumanNet 采样的约 **1000 小时** egocentric 人视频、以及约 **2 万小时** 真机训练的 LingBot 参照。论文报告验证损失上，**1000h egocentric 人视频** 可逼近或部分任务组上优于 **100h 真机** 初始化，并显著缩小与 **2 万小时** 真机参照的差距——用于支撑「egocentric 人视频在成本与规模上可作为机器人专有数据的部分替代」的主张（具体任务分组与损失曲线以论文图 6 为准）。
- **对 wiki 的映射：**
  - [Embodied Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md) — 数据规模、模态来源与下游性能的经验关系讨论语境
  - [Foundation Policy](../../wiki/concepts/foundation-policy.md) — 人类大规模视频作为 foundation policy 先验的数据论依据之一

## 关键术语

- **Human-centric video**：以人类物理交互为组织核心的剪辑，而非泛互联网被动观看型视频。
- **Robot-ready subset**：重定向误差与有效帧覆盖率阈值筛出的、更适合跨本体运动监督的子集。
- **Interaction-centric annotations**：围绕交互的字幕、运动语义、手/体几何等多层标注，而非仅类别标签。

## 关联 Wiki 页面

- [HumanNet（实体页）](../../wiki/entities/humannet.md)
- [VLA](../../wiki/methods/vla.md)
- [Imitation Learning](../../wiki/methods/imitation-learning.md)

## 当前提炼状态

- [x] 摘要级动机、管线三阶段与 robot-ready 准则
- [x] VLA 受控实验叙事（与真机小时对比的数量级）
- [ ] 细读全文附录后补充许可分布、子集划分比例与可复现训练配置细节
