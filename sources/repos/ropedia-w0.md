# Ropedia W0 官方仓库

> 来源归档（repo）

- **名称：** ropedia-w0
- **组织：** Ropedia
- **链接：** <https://github.com/Ropedia/ropedia-w0>
- **配套论文：** [Ropedia W0（arXiv:2610.26855）](../papers/ropedia_w0_arxiv_2610_26855.md)
- **项目页：** [Ropedia W0](../sites/ropedia-w0.md)
- **代码：** 部分开放：5B 推理入口、模型代码、18 个示例；5B pretrain/midtrain 权重和 Xp-Teleop 数据已另行开放；A14B、benchmark 微调权重、评测及后训练源码待发布
- **模型：** <https://huggingface.co/ropedia-ai/Ropedia-W0>
- **数据：** <https://huggingface.co/datasets/ropedia-ai/Xp-Teleop>
- **核对日期：** 2026-11-13
- **许可证：** 代码 Apache-2.0；模型权重 CC BY-NC 4.0；Xp-Teleop 数据 CC BY 4.0
- **一句话说明：** 跨本体第一视角运动-操作世界模型的官方 5B 推理实现仓库。

## 已核对运行入口

`scripts/run_single_case_5b.sh` 调用 `inference_5b.py`，后者转交 `ropedia_w0/cli.py:main`：先加载 Wan2.2-TI2V-5B 基座及 W0 的 camera/skeleton 条件分支与 checkpoint，再读取 81 帧窗口的首帧、相机位姿、骨架及 caption，构建 Plücker ray map，调用 `WanVideoPipeline` 输出 `*_pred.mp4` 等文件。

这套公开 CLI 默认为 **50 步离线生成**，与论文描述的 4 步蒸馏实时学生不同。仓库 README 要求至少 40 GB 显存 NVIDIA GPU，提供 PyTorch/CUDA 安装与基础模型下载说明。

## 对 wiki 的映射

- [Ropedia W0 实体页](../../wiki/entities/paper-ropedia-w0.md)
