---
type: entity
project_id: egoxperience
tags: [paper, spatial-intelligence, perception]
status: complete
updated: 2026-10-27
arxiv: "2610.26625"
project: "https://ropedia.github.io/EgoXperience/"
code: "https://github.com/Ropedia/EgoXperience"
related:
  - ./paper-ropedia-w0.md
  - ./paper-donic.md
sources:
  - ../../sources/papers/egoxperience.md
  - ../../sources/sites/egoxperience.md
  - ../../sources/repos/egoxperience.md
summary: "以统一世界坐标系联合重建第一视角视频中的相机、场景、手和物体。"
---

# EgoXperience：世界坐标系中的第一视角重建

## 一句话定义

以统一世界坐标系联合重建第一视角视频中的相机、场景、手和物体。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| 6DoF | Six Degrees of Freedom | 六自由度位姿 |
| MANO | Model with Articulated and Non-rigid defOrmations | 参数化手模型 |

## 核心原理

EgoX-Engine 通过几何验证的物体跟踪与 SLAM 对齐构造多模态监督；EgoX-Data-6M 包含 50 小时、约 540 万帧，EgoX 接收单目视频和首帧物体掩码，一次前向联合预测。

## 评测与结果

作者在 HOT3D、EgoX-Bench、HOI4D 和 H2O 上评估多项重建指标；论文报告相较所测最强基线，相机、手和深度误差分别下降 68.9%、65.6% 和 42.9%。

## 结论

以统一世界坐标系联合重建第一视角视频中的相机、场景、手和物体。

- 优先核对评测协议、输入条件与基线设置，再比较指标。
- 场景几何质量和感知误差会影响下游结果。
- 项目结果不能直接等同于真实机器人闭环任务表现。

## 源码运行时序图

不适用：官方仓库当前没有可运行的项目实现。

## 工程实践与局限

官方仓库目前为项目说明，代码、数据、基准和权重均列为待发布；不能据此认定可复现。

## 关联页面

- [Ropedia W0](./paper-ropedia-w0.md) — 跨本体第一视角世界模型。
- [DONIC](./paper-donic.md) — 从人类示范学习双手跟踪控制。

## 参考来源

- [论文归档](../../sources/papers/egoxperience.md)
- [项目页归档](../../sources/sites/egoxperience.md)
- [官方仓库归档](../../sources/repos/egoxperience.md)

## 推荐继续阅读

- [论文原文](https://arxiv.org/abs/2610.26625)
- [官方仓库](https://github.com/Ropedia/EgoXperience)
