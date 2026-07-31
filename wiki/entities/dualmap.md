---
type: entity
tags: [repo, semantic-mapping, open-vocabulary, sam, ros, navigation, clip, hkust-gz, hkust]
status: complete
updated: 2026-07-26
related:
  - ../queries/robot-perception-stack-selection-loop.md
  - ../queries/go2-3d-semantic-mapping-sam-pipeline.md
  - ./ovo-semantic-mapping.md
  - ./ov-sam3d.md
  - ./findanything.md
  - ./paper-segment-anything.md
  - ./paper-sam2.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ./orb-slam3.md
sources:
  - ../../sources/repos/dualmap.md
summary: "DualMap 是在线开放词汇语义建图系统：MobileCLIP + YOLO-World/MobileSAM/FastSAM 混合前端，双地图（全局抽象+局部具体），支持 ROS1/ROS2 与动态场景自然语言导航。"
---

# DualMap

**DualMap**（[Eku127/DualMap](https://github.com/Eku127/DualMap)，RAL 2025）是面向动态环境的 **在线开放词汇语义建图** 与自然语言导航系统。

## 一句话定义

用混合分割前端与 MobileCLIP 特征在线建图，维护 **全局抽象地图 + 局部具体地图**，支持自然语言查询目标并导航，且对场景变化可更新。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CLIP | Contrastive Language–Image Pretraining | 开放词汇查询的视觉语言特征 |
| SAM | Segment Anything Model | 二维分割；本系统可用 MobileSAM / FastSAM；基础模型见 [paper-segment-anything](./paper-segment-anything.md) |
| YOLO | You Only Look Once | 检测前端（YOLO-World 等） |
| ROS | Robot Operating System | 支持 ROS1 / ROS2 与 rosbag |
| OV | Open-Vocabulary | 不限定闭集类别列表的语义 |
| RAL | IEEE Robotics and Automation Letters | 论文发表 venue |

## 为什么重要

- 同时覆盖 **建图 + 语言导航 + 动态更新**，比「只投影 mask 到 PCD」更接近落地。
- 有 **ROS 模式**，便于后续接到 GO2 等移动平台（需自配相机/深度与时间同步）。
- 与 [OVO](./ovo-semantic-mapping.md)、[OV-SAM3D](./ov-sam3d.md)、[FindAnything](./findanything.md) 形成开放词汇语义选型池（FindAnything 仓仍待发布）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 香港科技大学（广州）（HKUST(GZ)）；香港科技大学（HKUST） |
| 代码 | <https://github.com/Eku127/DualMap> |
| 开源 | **已开源** |
| 论文 | arXiv:2506.01950（RAL 2025） |
## 核心原理

| 组件 | 说明 |
|------|------|
| **混合前端** | YOLO-World 类检测 + FastSAM/MobileSAM 等开放分割 |
| **特征** | MobileCLIP（v1 默认；文档亦述 v2） |
| **双地图** | 全局 abstract 候选选择；局部 concrete 精达目标 |
| **输入** | Dataset / ROS（含 bag）/ Record3d（iPhone） |

## 工程实践

1. 克隆时用 `--recurse-submodules` 拉 MobileCLIP。
2. 先 Dataset 或 rosbag 模式验证分割与查询，再上真机流。
3. 导航联调可参考上游 Habitat Data Collector（ROS2）文档。
4. 接到 GO2：几何位姿仍建议由 Point-LIO 提供；本仓作语义层，勿每帧硬跑最大模型。

## 局限与风险

- **非官方 GO2 一体栈**：传感器标定与 L1 投影需自建。
- 算力：官方实验叙事偏桌面 GPU（如 RTX 4090）；机载需换轻量权重与降频关键帧。
- 动态物体策略依赖其对象状态检查；静态占据层仍应单独管理。

## 关联页面

- [GO2 三维语义建图与 SAM 流水线](../queries/go2-3d-semantic-mapping-sam-pipeline.md)
- [OVO](./ovo-semantic-mapping.md)
- [OV-SAM3D](./ov-sam3d.md)
- [FindAnything](./findanything.md)
- [导航·SLAM 栈](../overview/navigation-slam-autonomy-stack.md)
- [ORB-SLAM3](./orb-slam3.md)

## 参考来源

- [sources/repos/dualmap.md](../../sources/repos/dualmap.md)
- 项目页：<https://eku127.github.io/DualMap/>
- arXiv：<https://arxiv.org/abs/2506.01950>

## 推荐继续阅读

- 上游 README：<https://github.com/Eku127/DualMap>
