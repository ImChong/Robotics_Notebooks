# T2-z-latent-optimization/T2-z

- **类型：** 官方代码仓库
- **仓库：** <https://github.com/T2-z-latent-optimization/T2-z>
- **项目页：** <https://t2-z-latent-optimization.github.io/>
- **论文：** [arXiv:2610.22364](https://arxiv.org/abs/2610.22364)
- **数据集：** <https://huggingface.co/datasets/huagailuowen/T2-z-datasets>
- **许可：** Apache-2.0
- **源码状态：** 已公开；数据与模型权重需单独下载（核验日期：2026-10-17）
- **知识节点：** [T2-z](../../wiki/entities/paper-t2-z-environment-latents.md)

## 运行入口

- 安装环境：Linux，README 推荐 Python 3.11 与 CUDA 12.8 对应的 PyTorch。
- 训练：`t2z/train.py`，配置位于 `configs/ours/` 与 `configs/baselines/`；仓库提供任务 launcher。
- 推理：`t2z/infer.py`，配置位于 `configs/inference/`；输入 checkpoint、context table、metadata 及 support/query 索引。
- 数据：使用仓库脚本下载并整理 HF 数据集；README 说明数据不随代码包提供。
- 权重：还需 Wan2.2-TI2V-5B 基座模型和 BWM 初始化 checkpoint；在线仿真另需 simulator dependencies。

## 关联资料

- [项目页归档](../sites/t2-z-latent-optimization.md)
- [论文摘录](../papers/t2_z_arxiv_2610_22364.md)
