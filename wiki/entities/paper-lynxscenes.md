---
type: entity
project_id: deeprobotics-lynxscenes
tags: [dataset, benchmark, wheel-legged, outdoor-navigation, perception, reconstruction, deep-robotics]
status: complete
updated: 2026-10-24
arxiv: "2610.18093"
project: "https://developer.deeprobotics.cn/research/articles/lynx-scenes/index.html"
related:
  - ../concepts/wheel-legged-quadruped.md
sources:
  - ../../sources/sites/deeprobotics-lynxscenes.md
summary: "LynxScenes（arXiv:2610.18093；云深处）：在 LYNX M20 上采集的室外轮足机器人基准，100K 个 30 秒片段、3.0M 标注帧、833.6 小时、3,294.7 km，统一感知 / 导航 / 想象 / 重建四类任务。"
---

# LynxScenes（arXiv:2610.18093）

**LynxScenes**（*LynxScenes: A Large-Scale Real-World Benchmark for Outdoor Wheel-Legged Robot Perception, Navigation, Imagination and Reconstruction*，[arXiv:2610.18093](https://arxiv.org/abs/2610.18093)，Jun Tian、Hao Wu、Ruo Chen、Yiyi Liao、Hongyi Lin、Jincheng Yu、Chao Li、Yingjie Niu；Deep Robotics、UIUC、HKUST、浙江大学、清华大学、MIT；[官方项目页](https://developer.deeprobotics.cn/research/articles/lynx-scenes/index.html)）。

## 一句话定义

**LynxScenes 是用真实 LYNX M20 同步采集环视、LiDAR 与运动状态的室外基准，用同一评测工具覆盖感知、导航、未来场景想象与重建。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LiDAR | Light Detection and Ranging | 激光雷达，M20 配 2 个 96 线 |
| LIO | LiDAR-Inertial Odometry | 运动状态来源之一 |
| BEV | Bird's-Eye View | 稠密鸟瞰标注 |

## 数据规模（项目页自报）

- 100K 个 30 秒片段、3.0M 标注帧、833.6 h 室外导航、3,294.7 km。
- 传感：4 路鱼眼（1920×1536、10 Hz）、2 个 96 线 LiDAR、6 维运动（IMU + LIO + 机器人状态）。
- 操作员按预设社交合规协议遥操，不安全或不一致示范在审核中剔除。

## 开源状态

Code 与 Dataset 均为 **SOON**（2026-10-24 核查），页面标注 ICLR 2027 审稿中。

## 结论

**目前只能读论文与项目页；数据与代码未发布，无法复核标注质量与评测工具。**

## 关联页面

- [DR-WM](./paper-dr-wm.md) — 云深处同批研究
- [云深处 rl_training](./deeprobotics-rl-training.md) — 同公司 RL 训练工程
- [wheel-legged-quadruped](../concepts/wheel-legged-quadruped.md)

## 参考来源

- [项目页归档](../../sources/sites/deeprobotics-lynxscenes.md)
- [arXiv 原文](https://arxiv.org/abs/2610.18093)
- [官方项目页](https://developer.deeprobotics.cn/research/articles/lynx-scenes/index.html)
- [云深处前沿探索](https://developer.deeprobotics.cn/research/)
