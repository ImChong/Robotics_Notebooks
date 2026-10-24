# DSL-Lab/StreamSplat

> 来源归档

- **标题：** StreamSplat 官方实现
- **类型：** repo / 3d-reconstruction / gaussian-splatting
- **代码：** <https://github.com/DSL-Lab/StreamSplat>
- **论文：** <https://arxiv.org/abs/2506.08862>（ICLR 2026）
- **项目页：** <https://streamsplat3d.github.io/>
- **权重：** <https://drive.google.com/drive/folders/1miH90t-hTwxfv0lc876_x1Xc7ZkoOFCY>
- **入库日期：** 2026-10-24
- **一句话说明：** 提供 StreamSplat 的静态编码器、动态解码器训练脚本、推理入口与预训练 checkpoint 下载地址。

## 仓库结构与入口

| 路径 / 文件 | 用途 |
|-------------|------|
| encoders/、model/ | 编码器与模型实现 |
| gaussian_renderer_dynamic/ | 动态高斯渲染相关代码 |
| train.py | 第一阶段：训练静态编码器 |
| train_decoder.py | 第二阶段：训练动态解码器 |
| splat_inference.py | 从帧和深度图执行推理 |
| preprocess_depth_davis.py | DAVIS 深度预处理脚本 |
| configs/、acc_configs/ | 数据、模型与分布式训练配置 |

## 快速路径（README）

创建仓库定义的 Conda 环境，安装 submodules/diff-gaussian-rasterization-orth，再运行 splat_inference.py；推理参数包括 checkpoint、RGB 帧目录和深度图目录。

深度图可依仓库说明使用 Depth Anything V2 生成；训练需准备 RealEstate10K、CO3Dv2、DAVIS 或 YouTube-VOS 数据并设置路径。预训练模型由项目主页链接的 Google Drive 提供。

## 对 wiki 的映射

- [StreamSplat 实体页](../../wiki/entities/paper-streamsplat-iclr-2026.md)
- [论文归档](../papers/streamsplat_iclr_2026.md)
- [项目页归档](../sites/streamsplat3d-github-io.md)
