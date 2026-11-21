# DIAL（xpeng-robotics/DIAL）

> 来源归档（官方源码仓 · Apache-2.0）

- **标题：** DIAL
- **类型：** repo
- **机构：** 小鹏机器人（XPENG Robotics）× HKU × UNC
- **代码仓：** <https://github.com/xpeng-robotics/DIAL>
- **项目页：** <https://xpeng-robotics.github.io/dial/>
- **论文：** <https://arxiv.org/abs/2603.29844>（[`sources/papers/dial_arxiv_2603_29844.md`](../papers/dial_arxiv_2603_29844.md)）
- **模型权重：** <https://huggingface.co/xpeng-robotics/DIAL_checkpoints>
- **分类：** VLA / 操作模型（基于 NVIDIA Isaac GR00T N1.5 代码栈改造）
- **入库日期：** 2026-09-06
- **核对范围：** 核对 `xpeng-robotics/DIAL@b611fed`（2026-06-02）仓库目录结构、`README.md`（全文 273 行）、`examples/train.sh`、`examples/eval.sh`、`examples/eval_loss.sh`、`scripts/inference_service_dial.py`、`scripts/simulation_service_dial_augPosRot.py`、`gr00t/eval/service.py`、`gr00t/eval/simulation_augPosRot.py`、`gr00t/model/policy_dial.py`（签名与类结构）。未在 GPU 环境运行训练或仿真。
- **通读：** 已完成
- **一句话说明：** 小鹏机器人开源的 DIAL 官方训练与仿真评测仓库，基于 Isaac GR00T N1.5 架构实现两阶段（decoupled golden bridge → end2end）训练与 ZMQ/msgpack 服务端–仿真客户端闭环评测。

## 开源状态（步骤 2.5）

- **已开源（Apache-2.0）：** 完整训练脚本（`examples/train.sh`、`examples/example_commands.sh`）、RoboCasa GR1 与 EgoDex 数据预处理脚本（`preprocessing/`）、在线仿真评测（`examples/eval.sh`）、离线动作/目标损失评测（`examples/eval_loss.sh`）及 Hugging Face 模型权重均已公开。

## 关键入口映射

| 阶段 | 入口脚本 | 核心模块 |
|------|----------|----------|
| 数据预处理 | `preprocessing/extract_and_visualize_3d-pos_6d-rot_from_gr1.py`、`aug_lerobot_data.py`、`convert_egodex_data_to_lerobot.py` | 为 GR1 补充 EEF 3D+6D 位姿 / 转换 EgoDex 到 LeRobot v2.0 |
| 两阶段训练 | `bash examples/train.sh <decoupled|end2end>` | `bridge_type=golden`（Stage 1）→ `bridge_type=end2end`（Stage 2） |
| 在线仿真评测 | `bash examples/eval.sh <model_path> <eval_type>` | `inference_service_dial.py`（`DialPolicy` + `RobotInferenceServer`）↔ `simulation_service_dial_augPosRot.py`（ZMQ msgpack） |
| 离线损失评测 | `bash examples/eval_loss.sh <model_path>` | 计算 action MSE、bridge goal loss 与 PCA 特征可视化 |

## 对 wiki 的映射

- [`wiki/entities/paper-dial-latent-world-vla.md`](../../wiki/entities/paper-dial-latent-world-vla.md)
