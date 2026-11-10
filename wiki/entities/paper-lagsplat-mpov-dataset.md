---
type: entity
project_id: lagsplat-mpov-dataset
arxiv: "2611.09959"
code: https://github.com/LouenPottier/lagsplat-mpov
project: https://louenpottier.github.io/xrdays.html
tags:
  - paper
  - dataset
  - deformable-object
  - gaussian-splatting
  - multi-view
status: complete
updated: 2026-11-10
related:
  - ./paper-deform360-deformable-visuotactile-dataset.md
  - ./paper-particlesplat.md
  - ./paper-sa-2406-10788-physically-embodied-gaussian-splatting-a-realtim.md
  - ../concepts/video-as-simulation.md
  - ../concepts/system-identification.md
  - ../concepts/state-estimation.md
sources:
  - ../../sources/papers/lagsplat-mpov-dataset_arxiv_2611_09959.md
original:
  - ../../sources/papers/lagsplat-mpov-dataset_arxiv_2611_09959.md
summary: "Pottier（Paris-Saclay / CEA List）数据集说明：同步 4 路 iPhone 视频（60 fps 量级、36 734 帧）、实测相机位姿与机械臂一维输入，记录约 23 cm 柔性条带受控振荡，并给出 LaGSplat 3D 基线（固定相机 PSNR 31.4 至 32.8 dB）。"
---

# 机械臂夹持柔性条带的四视角手机视频数据集（arXiv:2611.09959）

**论文：** *A four-view smartphone video dataset of a robot-held flexible strip, with camera poses and a 3D LaGSplat baseline*（Louen Pottier，Université Paris-Saclay, CEA, List；[arXiv:2611.09959](https://arxiv.org/abs/2611.09959)，v1，2026-10-26）。原文归档见 [sources 归档](../../sources/papers/lagsplat-mpov-dataset_arxiv_2611_09959.md)。

## 一句话定义

这是一篇 **2 页的数据集说明**：机械臂夹着约 23 cm 的柔性条带沿近一维路径猛拉，四部 iPhone 同步拍摄，附实测相机位姿、测得的输入 s(t)，以及用 LaGSplat 在其上做的 3D 重建与动力学基线（论文 Setup、Baseline 段）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LaGSplat | Latent Lagrangian Gaussian Splatting | 论文 [1]（arXiv:2608.16324）的方法：潜状态 q 既是学习到的耗散拉格朗日量的广义坐标，也是高斯溅射解码器的条件（本文摘要） |
| 3DGS | 3D Gaussian Splatting | 以显式高斯基元渲染场景 |
| PSNR | Peak Signal-to-Noise Ratio | 渲染图质量指标，越高越好 |
| BA | Bundle Adjustment | 光束法平差，联合优化相机参数与 3D 点 |
| PnP | Perspective-n-Point | 由 2D–3D 对应求相机位姿 |
| SIFT | Scale-Invariant Feature Transform | 局部特征，用于建手持相机的地图 |
| COLMAP | — | 运动恢复结构（SfM）工具，用于静态片段 |
| invex | invex function | 每个驻点都是全局极小的函数类；论文用“单极小”的势 V |

## 为什么重要

- 带可测输入的形变数据较少：本集同时提供多视角同步视频、相机位姿与机械臂输入 s(t)，可作为形变与动力学建模的真实多视角测试数据（作者自述的用途是支持 3D 拟合，见摘要“Fitting the decoder in 3D calls for synchronized multi-view footage”）。
- 作者交代对照：LaGSplat 原文的 3D 实验用 SHARP 渲染的合成视图，本数据集与结果不属于原文（摘要）。
- 采集只用消费级手机，无标定板，位姿全由场景求得（Camera geometry 段），复现门槛低（判断）。

## 核心原理与数据构成

```mermaid
flowchart LR
  A[机械臂夹持约 23 cm 柔性条带] --> B[沿近一维路径猛拉]
  B --> C[四部 iPhone 同步拍摄]
  C --> D[按 iphone1 时钟重采样 残差至多 8 ms]
  D --> E[相机位姿 BA 加 PnP 加 COLMAP]
  D --> F[state.csv 输入 s 与潜状态 q]
  E --> G[LaGSplat 3D 基线]
  F --> G
  G --> H[留出 60 s 上渲染 PSNR 与 rollout 误差]
```

数据构成（论文 Table 1 与正文）：

| 项 | 内容 |
|----|------|
| 视频 | scene_mobile/iphone{0,1,2,3}.mp4，每路 36 734 帧（612.8 s），1920×1080，60000/1001 fps，H.264，无音频 |
| 相机 | iphone0/1/2 固定且刚性相连，iphone0 机型不同；iphone3 手持，前 7 分钟移动后近乎静止 |
| 静态场景 | 物体全静止时 iphone3 另拍两段（6 303 与 1 722 帧，30 fps） |
| 几何 | 内参、畸变、位姿；3D 静态地标；世界系取 iphone0 相机系，长度单位约 0.38 m |
| 状态 | s(t)：夹爪沿路径弧长，在 iphone1 中跟踪，为实测输入；q(t)∈R²：基线编码的潜状态，仅供参考 |
| 区域 | zones.json：zone q（含条带）、zone s（手臂运动部分）两个有向盒 |
| 重定向就绪度 | 不适用：输入只是夹爪沿路径的一维弧长 s(t)，无机器人关节/动作轨迹，不能直接喂给机器人策略；训练输入形式为多视角视频 + 相机位姿 + s(t)，供形变重建与动力学拟合 |

## 基线实验与评测

基线（Baseline 段，在 LaGSplat [1] 上改三处）：机械臂作为输入，动力学 M q̈ = −∇V(q) − C q̇ + G s̈（常数 M、C，invex 势 V）；仿射解码器换成共享形变网络输出每个高斯的位移、尺度、旋转与颜色，J 在当前状态计算；背景用静态片段重建，仅 zone q 内高斯依赖 (s, q)。

基线结果（均在末 60 s 留出段，论文正文）：

| 指标 | 数值 |
|------|------|
| 固定相机渲染 PSNR（960 px） | 31.4 至 32.8 dB |
| 手持相机渲染 PSNR | 26.5 dB |
| 仅 zone q 投影内 PSNR | 固定 28.4 至 30.0 dB，手持 26.2 dB |
| 静态场景留出图 PSNR | 24.7 dB（逐颜色通道增益与偏置后） |
| rollout 归一化 RMS（除以 q 标准差） | 0.25 s：0.53；2 s：0.83 |

## 结论

**这份数据集的价值在于把同步多视角视频、场景求得的相机位姿与实测机械臂输入 s(t) 放进同一次录制，使带输入的形变动力学可在真实 3D 多视角数据上拟合；LaGSplat 基线说明可行，但只是单次录制上的起点。**

1. 四路 iPhone 各 36 734 帧（612.8 s），重采样残差至多 8 ms，无标定板即求得位姿。
2. 固定相机渲染 PSNR 31.4 至 32.8 dB，手持 26.5 dB；zone q 内更低（固定 28.4 至 30.0 dB）。
3. 动力学 rollout 归一化 RMS 从 0.25 s 的 0.53 升到 2 s 的 0.83，长时预测仍弱。
4. 手持相机仅 25 381 帧有位姿、标定误差 3 至 6 px，使用前需按帧筛选。

## 与其他工作对比

| 工作 | 数据 / 方法形态 | 与本数据集的区别 |
|------|-----------------|------------------|
| [Deform360](./paper-deform360-deformable-visuotactile-dataset.md) | 198 类可变形物体、41 环绕相机 + 双手触觉夹爪 | Deform360 面向规模与触觉；本集只有单一条带、单次录制，但附实测一维输入 s(t) |
| [ParticleSplat](./paper-particlesplat.md) | DLP 粒子扩展到 3DGS，报告下游策略成功率 | 本集基线不涉及策略学习，只评渲染 PSNR 与 rollout 误差 |
| [Physically Embodied Gaussian Splatting](./paper-sa-2406-10788-physically-embodied-gaussian-splatting-a-realtim.md) | 高斯–粒子双表示，支持预测与在线观测校正 | 本集基线用潜状态 q 的拉格朗日动力学条件化高斯形变，而非显式粒子 |

## 工程实践

- **同步：** 粗对齐用拍手与手臂猛拉序列，再按容器时间戳修正时钟率（iphone0 全程漂移 9 帧），统一重采样到 iphone1；使用时可直接读 frames.csv / sync.json 查每帧残差。
- **位姿：** 固定相机经联合光束法平差，自检三角化后投影到 iphone2 偏差 3 至 6 px；手持相机 36 734 帧中仅 25 381 帧有位姿，使用时需按帧筛选。
- **输入：** s(t) 为单视角像素量，换算物理量需自行标定（判断）。
- **复现：** 数据在 Zenodo（CC BY-NC 4.0，仅非商业）；训练代码仓库 README 称需 Python 3.13、CUDA GPU，重新训练得到相近而非相同数值，已发表解码器经过多次续训与最后的人工清理（README，静态读，未运行；论文未写这些细节）。
- 浏览器 3D 演示（论文称条带可随鼠标施力响应）本次未打开，未核实。

## 局限

- **论文性质：** 这是数据集说明 note，不是新算法论文，也不是机器人强化学习或策略学习工作；没有闭环控制、策略学习或任务成功率，机械臂输入是预先规定的猛拉运动，条带动力学也未用于控制。
- **论文自述局限：** 标定误差为几像素而非亚像素；handheld 段部分 iphone3 位姿错误；s(t) 仅在一个视角中以像素读取；iphone0 颜色与其余相机不同（Limitations 段）。
- **我的判断（非作者自述）：** 仅一条 612.8 s 的单次录制、单一条带与单一近一维输入，泛化到其他材质、多自由度操作未验证；基线只在同一录制的末 60 s 上留出，没有跨条带或跨会话评测；rollout 误差在 2 s 时为 0.83（已接近 q 标准差量级），长时预测较弱；基线对比对象仅有 LaGSplat 自身改动，无其他方法对照。
- 模型高斯数在 PDF 文字层显示为“105”，疑为 10^5 的上标丢失（推测），未核对原图。
- 被引的 LaGSplat 原文（arXiv:2608.16324）本次未读，其方法细节不在此页断言。

## 与「具身智能小站」条目的核对

- 小站称“同步四路手机视频、相机位姿与机械臂输入，用拉格朗日动力学和高斯表示重建柔性条带”：与论文一致。
- 小站称“移动相机位姿有缺失，标定也存在误差”：与论文一致（25 381 / 36 734 帧有位姿；标定误差 3 至 6 px，Limitations 段）。
- 小站称“可用于形变与动力学建模”：论文未直接这样陈述，属合理延伸；论文自述用途是 3D 拟合与基线。
- 小站题为“受控运动”：论文为单自由度夹爪猛拉，属受控输入，但并非多自由度机器人操作。

## 关联页面

- [Deform360 可变形物体视触觉数据集](./paper-deform360-deformable-visuotactile-dataset.md)
- [ParticleSplat](./paper-particlesplat.md)
- [Physically Embodied Gaussian Splatting](./paper-sa-2406-10788-physically-embodied-gaussian-splatting-a-realtim.md)
- [Video as Simulation](../concepts/video-as-simulation.md)
- [系统辨识](../concepts/system-identification.md)
- [状态估计](../concepts/state-estimation.md)

## 参考来源

- [sources/papers/lagsplat-mpov-dataset_arxiv_2611_09959.md](../../sources/papers/lagsplat-mpov-dataset_arxiv_2611_09959.md)
- [arXiv:2611.09959](https://arxiv.org/abs/2611.09959)
- 代码：<https://github.com/LouenPottier/lagsplat-mpov>

## 推荐继续阅读

- [arXiv PDF（v1）](https://arxiv.org/pdf/2611.09959v1)
- LaGSplat 原文：<https://arxiv.org/abs/2608.16324>
- 项目页：<https://louenpottier.github.io/xrdays.html>
