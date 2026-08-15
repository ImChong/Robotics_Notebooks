---
type: method
tags: [egocentric, hand-pose, 3d-vision, data-engine, manipulation, imitation-learning, vla, macrodata, wilor, hawor]
title: Macrodata Egocentric Hand-Action Pipeline
summary: "Macrodata Labs 工程博客给出的 RGB-only 开源配方：保守 WiLoR 检测 + 时序 HaWoR 手重建 + 窗口化 VGGT-Omega 度量相机轨迹与窄后处理，把 egocentric 视频变成世界系 21 关节度量手轨迹；HOT3D Action MPJPE 52.04 mm、81.23% 覆盖、15.53 FPS@H100。"
updated: 2026-08-15
status: complete
related:
  - ./wilor.md
  - ./egoscale.md
  - ./auto-labeling-pipelines.md
  - ./imitation-learning.md
  - ./vla.md
  - ../concepts/state-estimation.md
  - ../concepts/motion-retargeting.md
  - ../entities/paper-vidihand.md
  - ../entities/paper-hand-visibility-detector.md
  - ../entities/perceptron-egocentric.md
  - ../overview/ego-category-01-data-collection.md
  - ../overview/ego-category-02-human-to-robot.md
  - ../queries/dexterous-manipulation-data-pipeline.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/macrodata_egocentric_video_3d_hand_actions.md
  - ../../sources/sites/macrodata-co.md
  - ../../sources/repos/hawor.md
  - ../../sources/repos/wilor.md
  - ../../sources/papers/hand_visibility_detector_arxiv_2608_11574.md
---

# Macrodata Egocentric Hand-Action Pipeline

**Macrodata Labs**（[博客 2026-08-06](https://macrodata.co/blog/turning-egocentric-video-into-3d-hand-actions)）公开了一套 **RGB-only** 开源组件配方：把第一人称操作视频重建为 **共享世界系中的度量双手轨迹**，作为机器人策略的 **动作监督中间表示**（co-embodiment），并在 **HOT3D** 上用轨迹级 **Action MPJPE** 做可复现选型。

## 一句话定义

在 **≥75% 直接覆盖** 与 **≥15 FPS@H100** 约束下，用 **保守 WiLoR 检测 + 时序 HaWoR MANO 重建 + 窗口化 VGGT-Omega（depth-derived Sim(3) 拼接）+ 投影一致的窄后处理**，把 monocular egocentric RGB 转为每手 **21 度量 3D 关节** 的 1 秒相机相对 action chunk；选定配方 **52.04 mm Action MPJPE / 81.23% / 15.53 FPS**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPJPE | Mean Per-Joint Position Error | 关节位置平均欧氏误差；本文扩展为轨迹级 Action MPJPE |
| MANO | Hand Model with Articulated and Non-rigid deformations | 参数化手模型；HaWoR 预测形状/指姿后再解 21 关节 |
| RGB | Red-Green-Blue | 仅彩色单目视频；不用测量深度/LiDAR/立体/IMU |
| SLAM | Simultaneous Localization and Mapping | HaWoR 原用 DROID-SLAM；本文改为前馈 VGGT 窗 |
| Sim(3) | Similarity Transform in 3D | 旋转+平移+均匀尺度；用于对齐相邻 VGGT 窗 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略；用 1 秒 action chunk 对齐训练接口 |
| FPS | Frames Per Second | 端到端吞吐；门槛 15 FPS on H100 |
| HOT3D | Hands in Objects in 3D | 光学标记真值的 Aria/Quest egocentric 基准 |

## 为什么重要

- **填「像素多、动作少」的缺口：** 遥操作给干净 state–action，但难扩到 \(10^7\)–\(10^8\) h 量级 egocentric 语料；要把人视频当监督，必须先恢复 **度量手运动** 并去掉 **头动混入手动**。
- **把开源栈放到同一工程标尺上：** 不只报单帧手姿态，而用 **Action MPJPE + 覆盖 + 吞吐** 同时卡质量与成本（约 ≤2 H100·h / 视频小时）。
- **误差诊断可行动：** Shapley 归因显示剩余误差主要由 **相机系手（尤其腕深）** 贡献，而非相机轨迹——指导下一步该换检测/手模型还是几何模块。

## 主要技术路线

| 模块 | 要点 |
|------|------|
| **动作目标** | 每手 **21 度量 3D 关节**（腕 + 20 指），机器人无关；夹爪/[Motion Retargeting](../concepts/motion-retargeting.md) 表示可后派生 |
| **世界系融合** | [状态估计](../concepts/state-estimation.md) 式拆分：相机系手 × 度量相机轨迹 → 共享世界系，避免头动混入手动 |
| **VLA 切片** | 从帧 \(t\) 取未来 **1 秒**，再变回 \(t\) 相机系；**Action MPJPE** 不做 Procrustes，保留尺度与相对运动误差 |
| **检测** | 保守 [WiLoR](./wilor.md)（conf ≥ 0.75；≤4 帧间隙 IoU ≥ 0.20） |
| **手重建** | 时序 HaWoR MANO（16 帧 / 8 重叠），显著优于逐帧 WiLoR/HaMeR |
| **世界几何** | 窗口化 VGGT-Omega（200/40/416px）+ depth-derived **Sim(3)** 拼接，替换 DROID-SLAM+Metric3D |
| **后处理** | 仅投影一致修正（相机平移滤波、骨长/腕深）；拒绝直接关节平滑 |

### 流程总览

```mermaid
flowchart TB
  VID[Egocentric RGB video]
  DET[WiLoR detect<br/>conf ≥ 0.75<br/>≤4-frame gap IoU≥0.20]
  HAND[HaWoR temporal MANO<br/>16-frame / 8-overlap]
  CAM[VGGT-Omega windows<br/>200 / 40 overlap / 416px]
  ALIGN[Depth-derived Sim(3)<br/>+ linear blend]
  FUSE[World-space fusion<br/>R,t · p_cam]
  POST[Camera binomial filter<br/>bone-scale ±3.5%<br/>ray wrist-depth λ=0.2]
  OUT[Metric 21-joint<br/>hand-action trajectory]

  VID --> DET --> HAND
  VID --> CAM --> ALIGN
  HAND --> FUSE
  ALIGN --> FUSE --> POST --> OUT
```

### 选定开源配方（相对官方 HaWoR）

| 阶段 | HaWoR 原栈 | Macrodata 选定 |
|------|------------|----------------|
| 检测/跟踪 | WiLoR + BoT-SORT 类关联 | WiLoR + **保守短间隙**（0.75 / 4 帧 / IoU 0.20） |
| 相机系手 | 时序 HaWoR | **保留** HaWoR（显著优于逐帧 WiLoR/HaMeR） |
| 世界几何 | DROID-SLAM + Metric3D | **VGGT-Omega** 窗 + depth-derived **Sim(3)** |
| 缺口填充 | learned motion infiller | 导出保留显式缺失；评测用线性插值 |
| 后处理 | — | 相机平移轻滤波 + 骨长/腕深窄修正；**拒**关节平滑 |

## 工程实践

### 端到端数字（10× HOT3D Aria episode）

| 系统 | Action MPJPE | Direct coverage | FPS@H100 |
|------|--------------|-----------------|----------|
| 官方 HaWoR（infiller off） | 59.12 mm | 87.11% | 3.34 |
| 初始无对齐 VGGT + HaWoR | 90.73 mm | 89.14% | 25.01 |
| **选定配方** | **52.04 mm** | **81.23%** | **15.53** |

相对 HaWoR：**误差约 −12%**，吞吐约 **4.6×**。

### 关键旋钮（博客消融摘要）

- **几何窗：** 更长窗 / 更大重叠持续降 Action MPJPE；选定 200/40 仍清 15 FPS。
- **检测阈值：** 并非越严越好——过高覆盖掉线，过低引入坏框；0.75 为质量–覆盖拐点。
- **激进 tracker（ByteTrack/SAM2 等）：** 覆盖↑但难帧进入 HaWoR 后 3D 误差↑，前沿不优于保守规则。
- **后处理原则：** 只改 **不破坏 2D 投影** 的量（尺度/射线深度/相机平移）；直接关节平滑会「抹掉真运动」。

### 开源与产品边界

| 层级 | 状态（截至 2026-08-07） |
|------|------------------------|
| 博客公开的组件配方 | **可按上游复现**（WiLoR / [HaWoR](../../sources/repos/hawor.md) / VGGT-Omega 等） |
| Macrodata 端到端编排仓 / 专有检测 | **确认未开源**；官网 [Contact](https://macrodata.co/contact) / 免费样例标注 |
| 评测数据 | [HOT3D](https://huggingface.co/datasets/projectaria/hot3d)（非 Macrodata 资产） |

详情见 [sources/sites/macrodata-co.md](../../sources/sites/macrodata-co.md)。

## 局限与风险

- **覆盖代价：** 选定配方直接覆盖 **81.23%**，低于原 HaWoR 的 87%——质量换「少报难帧」，部署仍需处理显式缺失。
- **误差地板在手，不在相机：** 直接子集上相机系手贡献约 **32/39 mm**；深度轴约占 **43%**；大头动时相机误差仍会跳升。
- **HOT3D ≠ 客户域：** 手套、腕机、非穿戴者手等会使 WiLoR 失效；博客承认为此自研检测器。
- **非完整人→机闭环：** 重建轨迹可当 co-embodiment 监督；**本体 retarget / IK / 控制** 未评测。勿与 [EgoScale](./egoscale.md) 的「人数据预训练 VLA」或 [Perceptron Egocentric](../entities/perceptron-egocentric.md) 的「子任务语义分段」混为一谈。
- **复现风险：** 无官方 Macrodata 编排仓时，窗拼接与后处理细节需严格按博客消融复刻，否则易回到 ~90 mm 的断裂轨迹区。

## 关联页面

- [State Estimation](../concepts/state-estimation.md) — 手/相机分轨估计与世界系融合的感知语境
- [Motion Retargeting](../concepts/motion-retargeting.md) — 21 关节度量轨迹到机器人手/夹爪的下游接口
- [WiLoR](./wilor.md) — 检测与逐帧重建基线；本配方只用其 **检测头** 并提高置信门槛
- [HaWoR 源码归档](../../sources/repos/hawor.md) — 时序 MANO / 原世界重建基线
- [EgoScale](./egoscale.md) — egocentric 腕–手监督规模化进 VLA 的另一条主线
- [ViDiHand](../entities/paper-vidihand.md) — video diffusion 先验、无 detector 的 egocentric 双手 4D 对照
- [Hand Visibility Detector](../entities/paper-hand-visibility-detector.md) — 单帧逐关节可见性，可作检测后的按点门控
- [Auto-labeling Pipelines](./auto-labeling-pipelines.md) — 语义/成功标签数据引擎；本页侧重 **几何动作轨迹**
- [Perceptron Egocentric](../entities/perceptron-egocentric.md) — Macrodata **WGO** 子任务标注对照生态
- [Ego 数据采集](../overview/ego-category-01-data-collection.md) / [人→机器人](../overview/ego-category-02-human-to-robot.md)
- [灵巧操作数据管线 Query](../queries/dexterous-manipulation-data-pipeline.md)

## 参考来源

- [macrodata_egocentric_video_3d_hand_actions.md](../../sources/blogs/macrodata_egocentric_video_3d_hand_actions.md) — 本页主编译来源
- [macrodata-co.md](../../sources/sites/macrodata-co.md) — 公司页与开源边界核查
- [hawor.md](../../sources/repos/hawor.md) — HaWoR 上游
- [wilor.md](../../sources/repos/wilor.md) — WiLoR 上游
- 官方博客：<https://macrodata.co/blog/turning-egocentric-video-into-3d-hand-actions>

## 推荐继续阅读

- HaWoR 项目页：<https://hawor-project.github.io/>
- HOT3D 数据集：<https://huggingface.co/datasets/projectaria/hot3d>
- WiLoR 项目页：<https://rolpotamias.github.io/WiLoR/>
