# Kimodo-Policy: Learning Kinematic Action Models for Humanoid Loco-Manipulation — 官方源码

- **源码：** https://github.com/kimodo-policy/kimodo-policy
- **论文：** https://arxiv.org/abs/2611.11554v1
- **项目页：** https://kimodo-policy.github.io/
- **开放情况：** 官方仓库可访问；具体训练环境、预训练权重与依赖需要按仓库 README 单独配置。
- **核对范围：** GitHub README 与明确标注的源码/命令入口；尚未运行训练、评测或真机部署测试。
- **通读：** 未完成

## 运行入口与限制

开源包括 `train.py`、`model/kimodo_policy.py`、`model/modules/controlnet.py`、Arena 和 SIMPLE 的训练/评测脚本、模型权重及数据入口。需要区分独立 Conda 环境、NVIDIA SONIC ONNX 权重和仿真资源。

## 对 wiki 的映射

- [Kimodo-Policy：运动动作模型的视觉闭环移动操作](../../wiki/entities/paper-kimodo-policy.md)
