---
type: entity
project_id: streamsplat
tags: [paper, iclr2026, 3d-reconstruction, gaussian-splatting, dynamic-scene, online-reconstruction]
status: complete
updated: 2026-10-24
arxiv: "2506.08862"
venue: "ICLR 2026"
project: https://streamsplat3d.github.io/
code: https://github.com/DSL-Lab/StreamSplat
related:
  - ../entities/paper-sa-2602-11575-ready-go-real-to-sim-dynamic-3d-gaussian-splatti.md
  - ../entities/paper-rcl-2308-04079-3d-gaussian-splatting-for-real-time-radiance-fie.md
sources:
  - ../../sources/papers/streamsplat_iclr_2026.md
  - ../../sources/sites/streamsplat3d-github-io.md
  - ../../sources/repos/dsl_lab_streamsplat.md
summary: "StreamSplat（ICLR 2026）：从未标定视频流在线构建动态 3D Gaussian 表示，以概率式高斯采样、双向形变和自适应融合提升长序列重建稳定性。"
---

# StreamSplat：未标定视频流的在线动态 3D 重建

## 一句话定义

**StreamSplat** 是一个前馈式动态 3D Gaussian Splatting（3DGS）框架：逐帧处理未标定视频流，在线融合静态场景结构和动态形变，支持任意时刻的插值与新视角渲染。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| 3DGS | 3D Gaussian Splatting | 用三维高斯表示场景并进行可微渲染 |
| RGB-D | Red Green Blue - Depth | RGB 图像与深度图组成的模型输入 |
| DAVIS | Densely Annotated VIdeo Segmentation | 论文使用的动态视频基准之一 |
| VOS | Video Object Segmentation | YouTube-VOS 是论文使用的视频数据集之一 |

## 为什么重要

动态场景建模同时要处理相机运动、物体运动与时间跨度增长。StreamSplat 以逐帧前馈方式维护动态高斯表示，尝试避开传统方法的整段视频逐场景优化，使在线处理长流成为可能。论文讨论了机器人应用动机，但其贡献是通用视频重建方法；机器人导航或操控系统仍需另行验证感知、定位与控制闭环。

## 核心机制

### 1. 概率式静态高斯编码

每帧的 RGB 与深度信息进入静态编码器，输出规范空间内像素对齐的 3D Gaussians。论文用概率采样预测高斯位置，以刻画不确定性并降低位置预测落入局部最优的风险。仓库推理入口接收帧和深度图；深度可由 Depth Anything V2 预处理得到。

### 2. 双向动态形变

动态解码器在当前帧编码的高斯与历史传播的高斯之间建立双向形变关联。相较只沿单一方向传播，该机制旨在缓解长时间在线重建中的累积漂移，并让静态背景与运动物体保持一致的时空表示。

### 3. 自适应高斯融合

融合操作保留连续帧中持续存在的高斯，吸纳新进入视野的内容，并处理消失的高斯。组合后的表示支持在输入时刻之间渲染，用于视频重建、插值、深度估计和新视角合成。

## 流程总览

\`\`\`mermaid
flowchart LR
  V["未标定视频帧"]
  D["深度估计 / RGB-D"]
  E["静态编码器：概率式高斯采样"]
  F["动态解码器：双向形变"]
  G["自适应高斯融合"]
  R["动态 3DGS：任意时刻与视角渲染"]
  V --> D --> E --> F --> G --> R
\`\`\`

## 评测与指标

- **数据：** 静态场景评测包含 RealEstate10K 与 CO3Dv2；动态场景评测包含 DAVIS 与 YouTube-VOS。
- **结果读法：** 论文报告多个静态、动态基准上的重建质量，并称相对逐场景优化方法最高约 **1200× 加速**。该倍数是论文摘要中的总括主张，应回到对应基准、指标与硬件设置核验，不能直接等同于所有输入上的端到端实时吞吐。
- **消融：** 论文分析概率式位置采样、深度监督和双向形变等组件；源码仓库提供两阶段训练入口，便于检查实现与复现设置。

## 源码运行时序图

官方仓库提供训练与推理脚本，核心复现入口如下：

\`\`\`mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Prep as preprocess_depth_davis.py
    participant Depth as Depth Anything V2
    participant Train as train.py
    participant Decoder as train_decoder.py
    participant Infer as splat_inference.py
    participant Model as encoders/ 与 model/
    participant Render as gaussian_renderer_dynamic/
    Dev->>Prep: 准备 DAVIS 帧与深度
    Prep->>Depth: 生成伪深度图
    Dev->>Train: 第一阶段训练静态编码器
    Train->>Model: 编码 RGB-D 并学习规范高斯
    Dev->>Decoder: 第二阶段训练动态解码器
    Decoder->>Model: 学习双向形变与高斯融合
    Dev->>Infer: 输入帧、深度图和 checkpoint
    Infer->>Model: 逐帧预测动态高斯
    Model->>Render: 传递动态 3DGS 表示
    Render-->>Infer: 渲染 RGB / 深度 / 新视角结果
\`\`\`

推理路径依赖预计算深度图与 checkpoint；README 示例以 DAVIS 数据预处理为主，换用其他数据集时需按其格式准备对应输入。

## 工程实践

| 项目 | 说明 |
|------|------|
| **代码状态** | **已开源**；项目主页的 Code 链接指向 [DSL-Lab/StreamSplat](https://github.com/DSL-Lab/StreamSplat) |
| **模型权重** | 项目主页提供 Google Drive checkpoint 下载入口 |
| **训练** | train.py 训练静态编码器，train_decoder.py 训练动态解码器 |
| **推理** | splat_inference.py 使用 RGB 帧、深度图和 checkpoint |
| **数据准备** | DAVIS 有预处理脚本；其他数据集按 README 配置路径和深度图 |
| **机器人适配边界** | 输出是可渲染的场景表示，不含机器人位姿估计、地图规划或控制接口 |

## 结论

**StreamSplat 的关键贡献是把动态 3DGS 从逐场景离线优化推进到逐帧前馈在线重建；其长序列稳定性来自高斯位置采样、双向关联和自适应融合的组合。**

1. **部署读法：** 它的在线性来自模型逐帧推理；实际流水线仍需深度图、GPU 渲染与输入缓存，不能只看摘要加速倍数。
2. **结构读法：** 概率采样处理静态几何预测不确定性，双向形变和融合处理时间一致性及新旧内容更新。
3. **复现入口：** 从官方仓库的环境文件、DAVIS 深度预处理、两阶段训练脚本和 splat_inference.py 开始。
4. **适用边界：** 可作为动态场景表示与重建的参考；机器人导航、仿真或操控需要将其接入位姿估计、地图构建或规划链路后再评估。
5. **证据范围：** 论文在静态和动态视频数据集评测，不能据此推断真机机器人闭环性能。

## 关联页面

- [ReaDy-Go：面向视觉导航的动态 3DGS 仿真](paper-sa-2602-11575-ready-go-real-to-sim-dynamic-3d-gaussian-splatti.md)
- [3D Gaussian Splatting 原始论文](paper-rcl-2308-04079-3d-gaussian-splatting-for-real-time-radiance-fie.md)

## 参考来源

- [StreamSplat 论文归档](../../sources/papers/streamsplat_iclr_2026.md)
- [作者项目页及开源核查](../../sources/sites/streamsplat3d-github-io.md)
- [官方代码仓库归档](../../sources/repos/dsl_lab_streamsplat.md)

## 推荐继续阅读

- [StreamSplat 官方项目页](https://streamsplat3d.github.io/)
- [ICLR 2026 官方论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c822d05dcc00695ed6b63c9e97bb3449-Abstract-Conference.html)
