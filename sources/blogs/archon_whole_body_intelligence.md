# Whole-Body Intelligence: The Pretraining Path to Large Humanoid Models（Archon Robotics）

> 来源归档（blog / Archon Robotics 官方）

- **标题：** *Whole-Body Intelligence: The Pretraining Path to Large Humanoid Models*；中文版《全身智能：迈向人形基础模型》
- **类型：** 公司官方愿景 / 技术路线博文（无论文、无代码、无实验数字）
- **作者：** Archon Robotics（BibTeX 键 `archon2026wholebody`，author 字段为公司名，无个人署名）
- **原始链接：** 英文 <https://www.archon.tech/blog/whole-body-intelligence>；中文 <https://www.archon.tech/blog/whole-body-intelligence-cn>
- **发表日期：** 页面标注 **2026-07-13**（Published July 13, 2026；阅读时长 12 min）。同日为 RSS 2026 开幕日（悉尼），公司联合 OpenDriveLab 在会上做了三场演讲（见 [官网核查](../sites/archon-tech.md)）
- **入库日期：** 2026-10-10
- **抓取方式：** `curl` 读取服务端直出 HTML 并去标签；4 段演示视频懒加载，只取得 MP4 / 海报地址并查看海报
- **覆盖 wiki：** [Archon 全身智能（WBI）](../../wiki/entities/archon-whole-body-intelligence.md)、[源策未来（Archon Robotics）](../../wiki/entities/archon-robotics.md)

## 页面结构

英文页目录：01 Overview · 02 New Scaling Path · 03 What It Means · 04 Model Stack · 05 Human-Centric Data · 06 What Needs to Scale · 07 Pretraining（正文标题 *Hardware-Aware Pretraining*）· 08 Revealing Tasks · 09 Measuring Progress · 10 Closing。中文页目录：背景 · 为何现在做 · 全身智能定义 · 分层结构 · 全身数据 · 模型扩展方向 · 硬件适配预训练 · 关键任务 · 衡量指标 · 结语。

页首 4 段演示视频（`assets.archonrobotics.tech/blog/whole-body-intelligence/videos/*.mp4`）的标签：

| 英文 | 中文 | 海报所见（本库查看） |
|------|------|----------------------|
| Environment-Leveraged Manipulation | 环境借力操作 | 客厅场景，人形单腿动作，地上有毛绒玩具 |
| Long-Range Reachability | 大范围可达 | 厨房场景，人形双臂大幅外展 |
| Whole-Body Heavy-Object Manipulation | 全身协同重物操作 | 室外，人形弯腰扶起倒地的共享单车 |
| Narrow-Space Traversal | 窄距通行 | 厨房，人形在人与台面之间 |

机器人穿带 Archon 标志的 T 恤，页面 **未写机型、自主 / 遥操作方式、成功率或任务设置**。

## 关键主张（自报）

1. **定位：** 全身智能（WBI）是人形机器人的「预训练路径」，把任务、身体、接触、传感器、运动、失败和硬件限制作为一个系统来学习。目标不是取代全身控制，也不是「在人形上加一个 VLM」，而是做 **原生人形基础模型**，吸收人类数据、机器人数据、仿真、失败和部署反馈。中文版称这种模型为 **人形基础模型（Large Humanoid Model, LHM）**；官网首页用 **HFM（Humanoid Foundation Model）**。
2. **为什么需要新 scaling 路线：** 桌面操作有固定工作区、有限动作空间、稳定视角和短任务；人形的脚决定手能够到哪，躯干创造或消除操作空间，头部决定能看到什么，手会遮挡接触状态。移动与操作不是两个独立问题。问题从「能否控制身体」变成「能否预训练身体」。
3. **定义（原文）：** "Whole-body intelligence is the ability of a humanoid model to learn reusable full-body priors from heterogeneous human and robot experience, then use those priors to produce safe, adaptive, executable behavior in the real world." WBI 组织而非替代 WBC、BFM、VLA、agent 和世界模型。
4. **四层模型栈 S2 / S1 / S0.5 / S0：**
   - **S2 任务、语义与世界理解：** 语言、目标、场景理解、记忆、长程规划、任务分解、安全规则、重规划；接近 VLM agent、语义地图、mission controller。输出子目标。
   - **S1 原生人形基础模型（核心）：** 输入视觉、语言、本体感觉、触觉、历史、任务上下文；输出全身动作意图、action chunk 或 motion token。「不是加大动作头的机械臂 VLA」，要学视线、脚、躯干、手臂、手、接触与平衡的权衡。
   - **S0.5 运动生成 + BFM：** 接收紧凑目标、约束、motion token 或参考意图，生成可跟踪、可恢复的全身运动；通常不吃丰富视觉语言输入。闭环 BFM、运动生成框架归于此层。
   - **S0 全身跟踪器与控制器：** 面向硬件；跟踪参考运动，处理平衡、接触、关节与力限制、延迟与保护；可用本体感觉、参考运动、局部高程、接触状态或轻量感知，不负责全局任务理解。SONIC 类跟踪器归于此层。
   - 图中接口：Goal + language → S2 →（Subgoals）；Vision + body state → S1 →（Body priors）；Intent + constraints → S0.5 →（Reference motion）；Robot state → S0 →（Control targets）→ 实体机器人执行。
   - 「人形基础模型不是一个 checkpoint，而是一个复利循环」：人类数据、机器人 grounding 数据、仿真、失败、运行日志、评测与后训练都要回流。
5. **以人为中心的数据：** 遥操作数据贴合硬件但昂贵、窄、绑定本体；人类全身日常活动包含视线、平衡、可供性、身体间隙、接触与长程流程的密集先验，「不是机器人数据的廉价替代」。人类数据不能直接执行，正确路线不是脆弱的人到机器人重定向管线，而是 **异构预训练**，让模型学会哪些模式可迁移、哪些需重塑、哪些应拒绝。口号：「Human data gives scale. Robot data gives grounding. Failure data teaches recovery.」中文版补充数据形态：egocentric video、全身位姿、手部动作、IMU、语音和任务上下文。
6. **需要扩展的四个维度：** 模态（头部 / 腕部 / 掌心相机、深度、本体、触觉、力、音频、历史、局部接触）；身体（单臂 → 双臂 → 全身：蹲、侧步、跪、倚、边走边操作）；任务（开柜、整理房间、用工具、操作机器、搬运、可变形物、多分钟流程）；失败（抓空、打滑、门卡、遮挡、落脚不良、人为打断、硬件漂移）。数据飞轮：预训练 → 部署 → 观察失败 → 改数据 → 后训练 → 评测 → 再部署。
7. **硬件感知预训练（Hardware-Aware Pretraining）：** 电机精度、刚度、力控、手部自由度、触觉、掌心 / 头部相机、算力、延迟、电池与散热都是学习接口。过度耦合难迁移，完全硬件无关学不到物理细节，目标是学可迁移身体先验并尊重每台机器人的物理边界。很多「手」的问题其实是全身问题；人形操作不能简化为「机械臂 VLA + 人形底盘」。
8. **揭示问题的任务（正文举例）：** 走到垃圾桶、踩踏板、扔垃圾（引用 GR00T）；蹲下取桌下纸团，必要时先换视角或用脚拨出（中文版同）；打开低柜、弯腰取物、起身放到桌上（引用 HELIX）。这些任务无法干净拆成「导航 → 抓取 → 控制」。注意：第一、第三个例子的引用指向他人工作（NVIDIA GR00T、Figure Helix），纸团例子无引用；正文没有说这些是本公司的实验结果。
9. **衡量进展：** 新房间 / 光照 / 布局；同一可供性的新物体；失败后局部恢复；多子任务无人工复位；换手 / 换传感器 / 换机器人版本的适配成本；新任务需要多少真机数据；是否知道何时不安全、何时拒绝或求助。「价值不是最好的那条视频，而是学习、迁移、恢复和变得更安全的速度。」
10. **结语：** 「The next foundation model frontier for humanoids is not simply composing more skills. It is learning a body.」

## 参考文献（页面列出）

- 英文版 11 条：SONIC、AMS、BFM（bfm4humanoid.github.io）、π0.5、Reflect v1.0（Flexion）、DreamDojo、RISE、EgoScale、EgoHumanoid、GR00T、HELIX。其中 AMS、RISE、EgoHumanoid 链接到 OpenDriveLab 项目页。
- 中文版 12 条：HELIX、Reflect v1.0、SONIC、π0.5、Gemini Robotics、GR00T、GEN-1、SONIC、AMS、BFM、EgoScale、EgoHumanoid。
- 中文版正文写「SONIC、AMS（Agility Meets Stability，**Archon 方案原型**）」。AMS 即 arXiv 2511.17373（2025-11，早于公司成立），论文 HTML 中无 Archon 署名；「方案原型」是公司自述。

## 开源 / 实验状态（截至 2026-10-10）

| 项 | 结论 |
|----|------|
| 论文 | **无**（博文无 arXiv / 技术报告链接） |
| 代码 / 权重 / 数据 | **无**；GitHub `ArchonRobotics` 0 公开仓库，Hugging Face `ArchonRobotics` 0 模型 / 数据集 |
| 实验数字 | **无**（无成功率、基线、数据规模、模型参数量） |
| 演示 | 4 段短视频 + 首页完整视频 <https://youtu.be/b0h9oC8FhpU>，未说明机型与自主程度 |
| 计划 | 媒体称公司计划 2026 年内（下旬）发布首个开源人形基座模型（硬氪 2026-06-29，见 [新闻归档](archon_robotics_press.md)）；博文本身未提开源计划 |
