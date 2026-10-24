# StreamSplat 项目主页

> 来源归档（ingest）

- **标题：** StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams
- **类型：** project site
- **官方入口：** <https://streamsplat3d.github.io/>
- **论文：** <https://arxiv.org/abs/2506.08862>（ICLR 2026）
- **ICLR Proceedings：** <https://proceedings.iclr.cc/paper_files/paper/2026/hash/c822d05dcc00695ed6b63c9e97bb3449-Abstract-Conference.html>
- **代码：** <https://github.com/DSL-Lab/StreamSplat>
- **权重：** <https://drive.google.com/drive/folders/1miH90t-hTwxfv0lc876_x1Xc7ZkoOFCY>
- **作者：** Zike Wu、Qi Yan、Xuanyu Yi、Lele Wang、Renjie Liao
- **入库日期：** 2026-10-24
- **一句话说明：** 作者项目页展示由未标定视频流在线重建动态 3D Gaussian 场景，并提供论文、代码和 checkpoint 入口。

## 源码开放核查（2026-10-24）

| 项目 | 核查结论 |
|------|----------|
| **代码** | **已开源** — 项目主页 Code 链接指向 DSL-Lab/StreamSplat；仓库包含训练与推理脚本 |
| **权重** | **已提供下载入口** — 项目主页 Checkpoints 链接至 Google Drive |
| **数据** | 使用 RealEstate10K、CO3Dv2、DAVIS、YouTube-VOS 等基准；数据需按仓库 README 配置，项目页未提供统一数据包下载 |
| **可复现边界** | 推理示例输入 RGB 帧与预计算深度图；深度图可用 Depth Anything V2 生成。复现时还需安装仓库依赖并获取数据和权重 |

## 页面内容

项目页概述在线动态重建任务与方法，提供论文摘要、演示图、arXiv、代码和 checkpoint 链接。

## 对 wiki 的映射

- [StreamSplat 实体页](../../wiki/entities/paper-streamsplat-iclr-2026.md)
- [论文来源归档](../papers/streamsplat_iclr_2026.md)
- [仓库来源归档](../repos/dsl_lab_streamsplat.md)
