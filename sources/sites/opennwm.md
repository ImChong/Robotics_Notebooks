# OpenNWM 官方项目主页

> 来源归档（site）

- **标题：** OpenNWM: A Generalist Navigation World Model with Latent Action Pretraining
- **英文 / 中文：** A Generalist Navigation World Model with Latent Action Pretraining / 基于潜动作预训练的通用导航世界模型
- **项目页：** https://openspace-lab.github.io/OpenNWM/
- **论文 v1：** https://arxiv.org/html/2611.01844v1
- **官方代码：** https://github.com/OpenSpace-Lab/OpenNWM
- **模型：** https://huggingface.co/LittleB0sS/OpenNWM
- **数据：** https://huggingface.co/datasets/LittleB0sS/NavAnywhere
- **Go2 部署文档：** https://github.com/OpenSpace-Lab/OpenNWM/blob/main/GO2_DEPLOYMENT.md
- **核对范围：** 项目链接与 arXiv v1 对照、官方仓库 README 公开资源清单与 Go2 部署说明。
- **通读：** 已完成
- **入库日期：** 2026-11-04；**本次复核：** 2026-11-16

## 来源与可复现边界

- **已公开：** 训练/离线推理/评测代码、NavAnywhere 数据、LAM / Stage 1 / Stage 2 模型权重与文档。
- **尚未落地：** 官方 Go2 文档为 Jetson 部署计划，说明目前未验证机载性能，也没有 Go2 SDK/ROS2 闭环执行节点；官网视频和离线 Office-Go2 结果不能证明在线闭环性能。
- **数值口径：** 本页面引用的论文公式与实验只认 arXiv 2611.01844v1，不由最新仓库实验日志替换。

## 对 wiki 的映射

- [OpenNWM 详情](../../wiki/entities/paper-opennwm.md)
- [论文 v1 归档](../papers/opennwm_arxiv_2611_01844.md)
