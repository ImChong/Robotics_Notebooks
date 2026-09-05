---
type: query
tags: [dexterity, data-collection, teleoperation, simulation, robot-hand]
status: complete
updated: 2026-09-05
related:
  - ../entities/allegro-hand.md
  - ../entities/ruka-v2-hand.md
  - ../entities/midas-hand.md
  - ../entities/all-hands-up.md
  - ../entities/dexbench.md
  - ../entities/paper-dexholdem.md
  - ../entities/mimic-wearable-u1.md
  - ../entities/twindex.md
  - ../entities/paper-teledexter.md
  - ../entities/paper-nestdex.md
  - ../entities/paper-spd.md
  - ../entities/humantouch.md
  - ../entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md
  - ../comparisons/data-gloves-vs-vision-teleop.md
  - ../methods/behavior-cloning.md
  - ./demo-data-collection-guide.md
  - ../entities/paper-hand-visibility-detector.md
sources:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/teledexter_arxiv_2607_11481.md
  - ../../sources/papers/nestdex_arxiv_2608_13362.md
  - ../../sources/sites/aus-bot-nestdex.md
  - ../../sources/papers/spd_corl_2026.md
  - ../../sources/sites/spd-bot.md
  - ../../sources/sites/humantouch-xsparkai.md
  - ../../sources/sites/allhandsup-org.md
  - ../../sources/sites/x2robot-twindex.md
  - ../../sources/papers/hand_visibility_detector_arxiv_2608_11574.md
summary: "灵巧操作数据采集指南：介绍了如何利用 Shadow Hand、Allegro Hand 或低成本遥操作装置采集高质量、多模态的灵巧抓取与操作演示数据。"
---

# 灵巧操作数据采集指南

> **Query 产物**：本页由以下问题触发：「如何采集灵巧手操作的专家数据？有哪些主流的遥操作方案？」
> 综合来源：[Allegro Hand](../entities/allegro-hand.md)、[Demo Data Collection](./demo-data-collection-guide.md)

---

灵巧手（Dexterous Hand）的操作数据采集难度远高于普通的二指夹爪。由于自由度极高（通常 16-24 个），传统的中置式示教或简单的轨迹规划很难生成自然、丝滑的专家演示。目前主流的采集方案分为以下三类（详见 [数据手套 vs 视觉遥操作选型对比](../comparisons/data-gloves-vs-vision-teleop.md)）：

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Teleop | Teleoperation | 人遥操作机器人采集演示数据 |
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| RGB | Red-Green-Blue | 彩色图像通道，常与深度 (RGB-D) 配合 |

## 1. 基于视觉的动捕遥操作 (Vision-based Teleop)

这是目前最前沿且低成本的方案。
- **设备**：Leap Motion, Intel RealSense, 或 Meta Quest 摄像头。
- **原理**：利用手部追踪算法（如 [MediaPipe](../entities/mediapipe.md) 或专有引擎）实时估计人类操作者的指关节角度，并通过**重定向 (Retargeting)** 映射到灵巧手的 URDF 模型上。
- **优点**：无需佩戴繁琐设备，操作者手部无约束。
- **缺点**：视觉遮挡严重（例如手指重叠时）；缺乏力反馈，操作者很难感知抓握力度。需要按关节降权时，可在 21 点之上叠加 [Hand Visibility Detector](../entities/paper-hand-visibility-detector.md) 的可见概率（HInt 上系统评过，不是检测器整手分）。
- **代表项目**：AnyTeleop, DexCap；开源硬件侧 [RUKA-v2 Hand](../entities/ruka-v2-hand.md) 已集成 **AnyTeleop 向量重定向 + OpenTeach/Oculus VR** 遥操作管线；[MIDAS Hand](../entities/midas-hand.md) 提供 **MediaPipe 摄像头 + MANUS 手套** 双模态重定向与 **283 taxel 触觉同步流**（`midas-hand-org` 四仓库）。
- **固定运动学外骨骼（产业参考）**：[mimic wearable U1](../entities/mimic-wearable-u1.md) 用 **刚性连杆强制 M1 可达空间**，复制腕相机与指尖触觉布局，以 **零软件重定向** 采集中层数据——与视觉方案互补，见 [mimic 数据金字塔](../entities/mimic-hand-m1.md#数据金字塔中的位置)。自变量 [TwinDEX](../entities/twindex.md)（2026-09）把同一思想做成 **三指 7 主动外骨骼 ↔ 同构机械手**，宣称纯 robot-free 数据的 data-efficiency 与真机遥操作重叠、吞吐 **5.3×**；**未开源**，勿与同机构 XRZero-G0（VR+夹爪）混为一谈。
- **配对数据集参考**：[HRDexDB](../entities/hrdexdb-dataset.md) 采用 **XSens + MANUS 手套** 遥操 xArm6 + 多灵巧手，在 23 路同步相机下采集 **同物体人–机配对** 3D 轨迹与触觉（与纯视觉 teleop 的遮挡权衡不同）。

## 2. 穿戴式数据手套 (Data Gloves)

最成熟的工业级方案。
- **设备**：Manus VR, Shadow Glove, SenseGlove。
- **原理**：通过弯曲传感器或 IMU 阵列直接测量人类指节的弯曲度。
- **优点**：数据极其稳定，不受视觉遮挡影响。SenseGlove 等高级型号还能提供力反馈（Haptic Feedback），让操作者“摸”到虚拟物体。
- **缺点**：价格极其昂贵（数万美金）；设备校准繁琐。
- **触觉增强手套（人侧采数）**：[HumanTouch](../entities/humantouch.md) 在 MANUS **EMF** 手姿之上叠加全掌压阻触觉（约 **360** 点/手）与头/腕多视角 RGB，用姿态/历史感知标定与手套日检把「规模化」写成可追溯质控协议；初版约 100 h 宣称 HF 发布，**代码截至 2026-08-07 未列**。人机共用磁触觉迁移则见 [OSMO](../entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md)。

## 3. 仿真示教与自动生成 (Synthesized Data)

当真机采集太慢时，利用仿真环境“合成”数据。
- **方案 A：VR 交互**：人类佩戴 VR 头显在 MuJoCo 仿真环境里操纵灵巧手。
- **方案 B：RL 专家导出**：先用强化学习练出一个“完美策略”，再利用该策略生成轨迹作为模仿学习的负样本（Data Aggregation）。
- **方案 C：视觉重构**：从海量的人类操作视频（YouTube/Epic Kitchens）中，利用计算机视觉算法逆向推导出手的位姿序列。
- **方案 D：学习式接触执行层采数**：[TeleDexter](../entities/paper-teledexter.md) 用 MoCap 给出指尖+物体目标，仿真训好的 co-tracking 控制器在真机上完成 finger gaiting / 工具切换，从而采到运动学遥操作采不到的接触丰富示范（每任务约 50 条即可训 Diffusion Policy）。代价是 **物体专用策略 + 重型动捕**，且截至 2026-07-28 **未开源**。
- **方案 E：Copilot / 嵌套采数**：[NestDex](../entities/paper-nestdex.md)（arXiv:2608.13362）让操作员只控臂 + **1-DoF clutch**，可复用本体感觉内层手技能生成手指轨迹；完整示范再训**部署时不再依赖内层**的外层 visuomotor。相对同平台 AnyTeleop，六任务采数成功率 **100%** vs 三任务 **0%**。代价是每技能先采约 10 条重定向轨迹，且截至 2026-08-17 **未开源**。
- **方案 F：仿真 VR 预训练再真机短微调**：[SPD](../entities/paper-spd.md)（CoRL 2026 / [arXiv:2608.15917](https://arxiv.org/abs/2608.15917)）让操作员戴 Quest 在 MuJoCo 里直接控目标双臂灵巧手，5 人一周采 **75 h** on-embodiment 演示做扩散策略预训练；真机每任务再 **1–2 h** 微调。五项接触丰富任务都胜过从零 BC。代价是仿真接触必须调到「像真的」，且截至 2026-09-05 **代码与 spd-75h 未列下载**。

## 采集质量的 Checklist

- [ ] **时间戳对齐**：视频流、关节编码器、触觉传感器的数据必须毫秒级同步。
- [ ] **重定向精度**：检查人类指尖与灵巧手接触点的一致性，防止“虚空抓取”。
- [ ] **动作多样性**：同一个任务（如拿杯子）必须采集不同起始位置、不同朝向的数据，以防止模型过拟合。
- [ ] **多模态覆盖**：必须同时采集 RGB 图像、深度图和触觉力反馈，为后续的[多模态融合](./multimodal-fusion-tricks.md)做准备。

## 关联页面
- [Allegro Hand 实体](../entities/allegro-hand.md)
- [RUKA-v2 Hand 实体](../entities/ruka-v2-hand.md) — 全栈开源腱驱动 + VR 遥操作范例
- [MIDAS Hand 实体](../entities/midas-hand.md) — 直驱触觉开源手 + MediaPipe/MANUS 遥操作
- [MediaPipe 实体](../entities/mediapipe.md) — 低成本 21 点手部关键点感知框架
- [Hand Visibility Detector](../entities/paper-hand-visibility-detector.md) — 逐关节可见性，给视觉 teleop / 多视标注做按点门控
- [mimic wearable U1](../entities/mimic-wearable-u1.md) — 固定 M1 运动学的被动外骨骼中层采集
- [TwinDEX](../entities/twindex.md) — 三指外骨骼–机械手共设计；纯 robot-free、无软件 retarget（自变量，未开源）
- [Behavior Cloning](../methods/behavior-cloning.md)
- [TeleDexter（论文实体）](../entities/paper-teledexter.md) — co-tracking 灵巧遥操作数据引擎
- [SPD（论文实体）](../entities/paper-spd.md) — 仿真 VR 75 h 预训练 + 真机 1–2 h 微调（CoRL 2026；代码待发布）
- [NestDex（论文实体）](../entities/paper-nestdex.md) — clutch copilot 嵌套采数，部署卸内层（arXiv:2608.13362；未开源）
- [HumanTouch](../entities/humantouch.md) — 人手全掌压阻触觉 + EMF 手姿多模态采数（数据待发）
- [OSMO 触觉手套](../entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md) — 人机共用开源磁触觉对照
- [多模态融合技巧](./multimodal-fusion-tricks.md)
- [操作演示数据采集总指南](./demo-data-collection-guide.md)
- [All Hands Up](../entities/all-hands-up.md) — RLWRLD 把高背驱、力矩可当学习信号的手标成 Type 2 采数手
- [DexBench](../entities/dexbench.md) — Regime → 传感/标注优先级；本指南补「先定测什么再定采什么」
- [DexHoldem](../entities/paper-dexholdem.md) — 1,470 条 ShadowHand 扑克示范；Vive 遥操作，失败尝试不进发布集

## 参考来源
- Qin, B., et al. (2023). *AnyTeleop: A Unified and General Framework for Bimanual Dexterous Teleoperation*.
- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md)
- [sources/papers/teledexter_arxiv_2607_11481.md](../../sources/papers/teledexter_arxiv_2607_11481.md)
- [sources/papers/nestdex_arxiv_2608_13362.md](../../sources/papers/nestdex_arxiv_2608_13362.md) — NestDex copilot 采数
- [SPD 论文归档](../../sources/papers/spd_corl_2026.md) — 仿真 VR 灵巧手预训练
- [sources/sites/aus-bot-nestdex.md](../../sources/sites/aus-bot-nestdex.md) — 项目页开源核查
- [sources/sites/humantouch-xsparkai.md](../../sources/sites/humantouch-xsparkai.md)
- [All Hands Up 站点归档](../../sources/sites/allhandsup-org.md)
- [x2robot-twindex.md](../../sources/sites/x2robot-twindex.md) — TwinDEX 无本体三指共设计
- [hand_visibility_detector_arxiv_2608_11574.md](../../sources/papers/hand_visibility_detector_arxiv_2608_11574.md) — 视觉 teleop 遮挡时的逐关节可见性
