# RTNav 官方代码仓库

> 来源归档（repo）

- **标题：** RTNav: Towards Real-Time Zero-Shot Object Navigation
- **类型：** repo
- **仿真与评测仓库：** <https://github.com/generalroboticslab/RTNav>
- **真实机器人部署仓库：** <https://github.com/generalroboticslab/RTNav-RealWorld>
- **arXiv：** <https://arxiv.org/abs/2608.26496>
- **项目页：** <https://generalroboticslab.com/RTNav>
- **机构：** 杜克大学（Duke University）General Robotics Lab
- **入库日期：** 2026-10-24
- **一句话说明：** 仿真仓库实现异步 RTNav agent 与统一实时评测框架；独立部署仓库提供 Stretch 3 / Jetson AGX Thor 适配。
- **沉淀到 wiki：** [`wiki/entities/paper-rtnav.md`](../../wiki/entities/paper-rtnav.md)

## 仓库内容与运行入口

| 仓库 | 入口 / 内容 |
|------|-------------|
| `generalroboticslab/RTNav` | `agents/` 含 RTNav 与 VLFM、L3MVN、TriHelper、GAMap、OpenFMNav、BeliefMapNav 基线；`envs/`、`docker/` 与 ROS 2 接口用于 Habitat 实时评测。 |
| `generalroboticslab/RTNav-RealWorld` | `src/env/launch.py` 启动 ROS 1→ROS 2 观测桥；`src/agent/rtnav_runner.py` 启动真实机器人 agent；子模块固定上游 RTNav 版本。 |

## 开源状态（核查 2026-10-24）

- **已开源：** 两个官方仓库均为公开仓库；仿真仓库 README 含 Docker build 与评测命令，真实机器人仓库含硬件部署步骤。
- **复现注意：** 数据集与模型权重需按仓库文档另行准备；论文实时结果使用 RTX A6000，需报告测试硬件并在同一硬件上对照基线。