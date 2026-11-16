# OpenNWM 官方代码与模型入口

> 来源归档（repo）

- **标题：** OpenNWM: A Generalist Navigation World Model with Latent Action Pretraining / 基于潜动作预训练的通用导航世界模型
- **官方仓库：** https://github.com/OpenSpace-Lab/OpenNWM
- **项目页：** https://openspace-lab.github.io/OpenNWM/
- **论文 v1：** https://arxiv.org/html/2611.01844v1
- **模型（Hugging Face）：** https://huggingface.co/LittleB0sS/OpenNWM
- **模型（ModelScope）：** https://modelscope.cn/models/LittleBoss/OpenNWM
- **NavAnywhere（Hugging Face）：** https://huggingface.co/datasets/LittleB0sS/NavAnywhere
- **NavAnywhere（ModelScope）：** https://modelscope.cn/datasets/LittleBoss/NavAnywhere
- **Go2 文档：** https://github.com/OpenSpace-Lab/OpenNWM/blob/main/GO2_DEPLOYMENT.md
- **核对范围：** 仓库根 README.md、run.py、lam/README.md、lam/run.py、nwm/README.md、nwm/run.py、nwm/train.py、nwm/planning_eval.py、nwm/conf/two_stage/go2_adapt.yaml 与 GO2_DEPLOYMENT.md；运行入口已按源码核对，未实际训练或部署。
- **通读：** 已完成
- **入库日期：** 2026-11-04；**本次更新：** 2026-11-16

## 开放内容（官方 README，2026-11-16）

- 代码、模型、NavAnywhere 数据与离线训练/评测说明已公开；公开压缩数据集约 1.93 TB；TartanDrive 需按 NWM 说明另行下载。
- 三个独立权重：LAM `lam-nav15-pixel-action-step60k.ckpt`、Stage 1 `opennwm-nav15-stage1-pretrain60k.pth.tar`、Stage 2 `opennwm-nav15-stage2-joint100k.pth.tar`。
- 根 `run.py` 把命令按环境分发到 `lam/run.py` 或 `nwm/run.py`；LAM 后端运行 `navigation_main.py`；NWM 后端分别调用 `train.py`、`prepare.py`、`isolated_nwm_infer.py`、`isolated_nwm_eval.py`、`planning_eval.py`。
- 仅当有 Stage 2 real-action adapter checkpoint 时，才能按官方条件运行真实 waypoint 动作的推理 / 规划。
- 依赖 Linux、Conda、CUDA 12.4 兼容 GPU；LAM 与 NWM 使用各自 Python 3.10 环境，不可直接推断 Jetson aarch64 包兼容性。

## Go2 支持的明确边界

- `GO2_DEPLOYMENT.md` 题为“Go2 EDU + Jetson Deployment Plan”，明确 Jetson 配置与性能尚未确认，**没有经验证的机载闭环运行结果**。
- 存在 `nwm/conf/two_stage/go2_adapt.yaml` 离线 finetune 配方与 `nwm/planning_eval.py` 离线规划，但缺 Go2 SDK/ROS2 执行节点、定位/障碍停机与安全控制桥。
- `go2_adapt` 要求兼容的真实动作 checkpoint（不能直接由 LAM、Stage 1 替代）；需要按实测位姿准备 Go2 数据。
- 论文 v1 Office-Go2 ATE 2.89 / RPE 0.65 属离线评测，不能当真机闭环指标。

## 映射

- [论文详情](../../wiki/entities/paper-opennwm.md)
- [论文 v1 公式与实验归档](../papers/opennwm_arxiv_2611_01844.md)
