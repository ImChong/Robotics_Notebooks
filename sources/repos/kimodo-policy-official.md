# Kimodo-Policy: Learning Kinematic Action Models for Humanoid Loco-Manipulation — 官方源码

- **源码：** https://github.com/kimodo-policy/kimodo-policy
- **论文：** https://arxiv.org/abs/2611.11554v1
- **项目页：** https://kimodo-policy.github.io/
- **开放情况：** 官方仓库可访问；具体训练环境、预训练权重与依赖需要按仓库 README 单独配置。
- **核对范围：** 2026-11-13 阅读官方 README 的安装、训练、Arena/SIMPLE 评测及真机采集/部署章节；对照 `train.py`、`model/kimodo_policy.py`、`model/modules/controlnet.py`、`evaluation/humanoidarena_eval_multi_task_sonic.sh`、`evaluation/humanoidarena_eval_signal_task_sonic.sh`、`evaluation/humanoidarena_server.py`、`real-world/deploy/{deploy,tunnel,sonic}.sh` 的运行入口。未在本环境实际安装、训练、评测或上机。
- **通读：** 已完成官方 README 对应的发布与复现章节；大型源码仓仅核对上述关键入口，不声称全仓源码通读

## 运行入口与限制

开源包括 `train.py`、`model/kimodo_policy.py`、`model/modules/controlnet.py`、Arena 和 SIMPLE 的训练/评测脚本、模型权重及数据入口。需要区分独立 Conda 环境、NVIDIA SONIC ONNX 权重和仿真资源。

## 开源与复现边界（2026-11-13）

- **已发布（README Roadmap 勾选）：** Kimodo-Policy 预训练、HumanoidArena 训练/评测、SIMPLE 训练/评测、真机采集/训练/部署；BiGym 2.0 和 HumanoidToolBench 标注待发布。
- **发布物：** [模型与任务检查点](https://huggingface.co/YunhengWang/kimodo-policy)、[数据和评测结果](https://huggingface.co/datasets/YunhengWang/kimodo-policy)；真机八项任务共 236 条演示（33+29+25+24+30+38+28+29），并公开对应任务检查点及数据。
- **三套环境：** `kimodo-policy` (Python 3.10) 训练/模型服务；`unitree_sim_env` (Python 3.11，Isaac Sim 5.0.0、Isaac Lab 2.2.0) 负责 HumanoidArena；`SIMPLE/.venv` (Python 3.10，MuJoCo 3.3.6、Isaac Sim 4.5.0) 负责 SIMPLE。
- **评测时序：** `evaluation/humanoidarena_eval_multi_task_sonic.sh` 按任务调用 `evaluation/humanoidarena_eval_signal_task_sonic.sh`，仿真通过 `evaluation/humanoidarena_server.py` 获取动作块；多任务训练脚本通过 `accelerate launch train.py --config YAML` 训练。
- **真机时序：** `real-world/deploy/deploy.sh` 委托 `deploy_cli.py`，先 GPU 端 `server`，再工作站 `tunnel`、`sonic`、`check`、`run`。PICO 主要用于遥操作采集，推理需真实相机、G1 状态与 SONIC；带校验清单的工作站运行时资源包另行下载。
- **安全/复现：** 作者要求现场支撑、急停与控制器安全确认；`check` 模式不发布动作。README 的安装检查和 `--dry-run` 仅验证资源/命令，不能证明真机效果。完整 GPU/硬件实验未执行；对跨控制器表现不作推断。
- **参考：** [官方 README](https://github.com/kimodo-policy/kimodo-policy#readme)、[真机工作流](https://github.com/kimodo-policy/kimodo-policy/blob/main/real-world/README.md)、[部署指导](https://github.com/kimodo-policy/kimodo-policy/blob/main/real-world/deploy/start.md)。

## 对 wiki 的映射

- [Kimodo-Policy：运动动作模型的视觉闭环移动操作](../../wiki/entities/paper-kimodo-policy.md)
