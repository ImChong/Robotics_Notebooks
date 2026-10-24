# StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams（ICLR 2026）

> 来源归档（ingest）

- **标题：** StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams
- **类型：** paper / 3d-reconstruction / gaussian-splatting / dynamic-scene
- **会议：** ICLR 2026
- **ICLR Proceedings：** <https://proceedings.iclr.cc/paper_files/paper/2026/hash/c822d05dcc00695ed6b63c9e97bb3449-Abstract-Conference.html>
- **arXiv：** <https://arxiv.org/abs/2506.08862>
- **项目页：** <https://streamsplat3d.github.io/>
- **代码：** <https://github.com/DSL-Lab/StreamSplat>
- **权重：** <https://drive.google.com/drive/folders/1miH90t-hTwxfv0lc876_x1Xc7ZkoOFCY>
- **作者：** Zike Wu、Qi Yan、Xuanyu Yi、Lele Wang、Renjie Liao
- **机构：** University of British Columbia、Vector Institute for AI、Nanyang Technological University、Canada CIFAR AI Chair
- **入库日期：** 2026-10-24
- **一句话说明：** 将未标定视频流逐帧转为动态 3D Gaussian Splatting 表示，以前馈方式支持任意长度视频的在线重建、插值和新视角合成。

## 摘要级要点

- **问题：** 传统动态场景重建常需要整段视频和逐场景优化；StreamSplat 面向稀疏观测、低延迟和长视频在线处理。
- **概率式静态编码：** 将输入帧编码为规范空间中的像素对齐 3D Gaussians，并对位置进行概率采样，缓解不确定输入下的位置预测局部最优。
- **双向动态建模：** 动态解码器预测双向形变场，将当前帧与历史高斯关联，减轻长序列误差积累。
- **自适应高斯融合：** 延续跨帧持续存在的高斯，同时整合新出现或消失的场景内容。
- **评测主张：** 论文报告在静态和动态基准上取得有竞争力的重建结果，并称相对逐场景优化方法最高约 1200× 加速；速度与质量应结合论文具体基准和设置解读。
- **边界：** 论文讨论通用视频重建及机器人等潜在应用，不是机器人专用定位或闭环导航系统。

## 核心论文摘录（MVP）

### 1. 在线重建框架
- **来源：** ICLR Proceedings 摘要、论文 §1–3
- **要点：** 输入是未标定的视频流；每帧先由静态编码器预测规范空间高斯，再由动态解码器预测帧间形变并融合，生成的动态表示可在任意时刻和视角渲染。
- **对 wiki 的映射：** [StreamSplat 实体页](../../wiki/entities/paper-streamsplat-iclr-2026.md)

### 2. 概率式高斯位置预测
- **来源：** 论文 §3.1、§4.4
- **要点：** 静态编码器使用概率采样预测高斯位置；论文消融报告该设计改善关键帧重建，并讨论确定性位置预测更易陷入局部最优。
- **对 wiki 的映射：** [StreamSplat 实体页](../../wiki/entities/paper-streamsplat-iclr-2026.md)

### 3. 双向形变与自适应融合
- **来源：** 论文 §3.2–3.3、§4
- **要点：** 双向形变建立当前与历史高斯的对应，自适应融合处理持续、新增及消失内容，目标是稳定长序列建模并支持中间时刻渲染。
- **对 wiki 的映射：** [StreamSplat 实体页](../../wiki/entities/paper-streamsplat-iclr-2026.md)

## BibTeX

@inproceedings{wu2026streamsplat,
  title     = {StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams},
  author    = {Wu, Zike and Yan, Qi and Yi, Xuanyu and Wang, Lele and Liao, Renjie},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2026/hash/c822d05dcc00695ed6b63c9e97bb3449-Abstract-Conference.html}
}

## 对 wiki 的映射

- 主实体页：[StreamSplat](../../wiki/entities/paper-streamsplat-iclr-2026.md)
- 项目页归档：[streamsplat3d.github.io](../sites/streamsplat3d-github-io.md)
- 代码仓库归档：[DSL-Lab/StreamSplat](../repos/dsl_lab_streamsplat.md)
