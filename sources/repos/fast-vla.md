# Garfield-kh/FastVLA

- **类型：** 官方代码仓库
- **仓库：** <https://github.com/Garfield-kh/FastVLA>
- **项目页：** <https://garfield-kh.github.io/FastVLA/>
- **模型：** <https://huggingface.co/kehong/FastVLA-weights>
- **论文：** [arXiv:2610.21389](https://arxiv.org/abs/2610.21389)
- **源码状态：** 训练/推理代码、RoboTwin 与 LIBERO 评测客户端及两类 checkpoint 已公开（核验日期：2026-10-19）
- **知识节点：** [Fast-VLA](../../wiki/entities/paper-fast-vla-once-per-episode.md)

## README 入口

- 安装：Python 3.11、PyTorch CUDA 12.8；依赖 OpenDM 固定提交。
- 组件：VLM task-memory bank 构建、在线 actor、policy server、RoboTwin 2.0 / LIBERO / LIBERO-Plus 客户端。
- 在线常驻显存/内存指标不包含 VLM 初次预填充的峰值；复现对比时需按项目页的计量口径。

## 关联资料

- [项目页归档](../sites/fast-vla.md)
- [论文摘录](../papers/fast_vla_arxiv_2610_21389.md)
