# MuJoCo Playground 1.0 官方项目页

> 一手资料归档：本页记录 2026 年 11 月发布的 **1.0 系列**；2025 年 RSS 论文与 JAX/MJX 研究版作为历史版本，不混写。

- **标题：** MuJoCo Playground 1.0
- **类型：** 项目主页 / 官方文档
- **维护方：** Google DeepMind Robotics Simulation Team
- **项目页：** https://playground.mujoco.org/
- **官方文档：** https://reaf-sim.readthedocs.io/en/latest/playground/index.html
- **PyPI 版本：** https://pypi.org/project/playground/ （1.0.0：2026-11-06；1.0.1：2026-11-09，核对日最新）
- **代码：** https://github.com/google-deepmind/mujoco_playground （**已开源**，Apache-2.0）
- **旧版论文：** https://arxiv.org/abs/2502.08844 （RSS 2025 / JAX，不应等同于 1.0 技术报告）
- **核对日期：** 2026-11-10
- **核对范围：** 官方主页完整文字、What's New in 1.0 对照、Quickstart CPU/CUDA12/CUDA13、五类工作流、机器人清单；官方 1.0 在线文档 Overview 和 PyPI 版本/依赖说明；官方仓库 README 技术栈与安装、运行入口另见 repo 归档。
- **通读：** 已完成
- **主实体：** [MuJoCo Playground](../../wiki/entities/mujoco-playground.md)
- **源码归档：** [mujoco_playground](../repos/mujoco_playground.md)

## 官方 0.2 → 1.0 变更核对

| 维度 | 0.2（RSS 2025 路线） | 1.0 |
|---|---|---|
| 训练框架 | JAX | PyTorch |
| 物理 | MJX（JAX/Warp backend） | MuJoCo CPU / MuJoCo Warp |
| 环境 | 以 RL 仿真任务为主 | 机器人及传感器设备层和任务层分离，统一仿真与真机运行 |
| 训练 | 经典环境、MJX/Brax/JAX 工具链 | RSL-RL（状态 / 视觉输入）、GPU 批量任务 |
| 交互与工程 | 仿真、训练展示 | MuJoCo Studio、遥操作、MCAP、系统辨识、部署与 agentic VLM |

官方页面明确将核心技术栈写成「Framework: JAX → PyTorch」，不是说 **JAX 软件本身被废弃**。官方当前 README 明确保留 `rss2025` 目录供旧 JAX/MJX 环境使用。

## 工作流与机器人

- **Whole-body motion tracking**：Unitree G1（官方列 29 DoF）参考运动跟踪，RSL-RL 并行训练。
- **System identification**：`mujoco.sysid` 依照硬件日志调整仿真关节动力学。
- **Sim / real teleoperation**：SO-101 leader/follower、脚踏开关限定 episode、MCAP 日志。
- **Zero-shot sim-to-real**：同一任务层与策略，通过 `--mode=sim` 与 `--mode=real` 切换（须具备硬件连接、校准、安全措施）。
- **Agentic manipulation**：SO-101 与 i2rt YAM 的 VLM 操作案例；仓库内有执行生成 Python 的风险提示。
- **机器人清单**：SO-101 和 YAM Pro 标注 `sim real`；WidowX AI 与 Unitree G1 标注 `sim`。不能把所有机器人称为开箱即用真机。

## 最小官方例子

CPU（CartpoleBalance；页面称约 1 分钟，实际耗时依设备而异）：

```bash
uvx --default-index https://pypi.org/simple --python 3.13 --from "playground[rl]>=1.0" \
  python -m mujoco_playground.rl.rsl_rl.train \
  --env=CartpoleBalance --logdir=/tmp/rslrl_logs/cartpole_balance \
  --overrides='{"nworld": 64}'
```

官方 CUDA 12 / CUDA 13 另有 Warp 和 PyTorch wheel 索引差异，勿照搬旧版 `jax[cuda12]` 安装流程。

## 范围与证据边界

- 项目页与当前 GitHub README/源码、PyPI 包均已开放，1.0 **不是「Coming Soon」**。
- 1.0 官方页面描述功能路径，并**未提供对比 Isaac Lab / mjlab 的统一性能基准**；本归档不据此声称训练一定更快。
- 未下载训练权重、未执行本机/真机 benchmark；页面里的零样本迁移是官方示例，不保证所有机器人、所有任务成功。
- **版权 / 引用：** 1.0 软件引用 `@software{mujoco_playground_2026,...}`；RSS 2025 论文另行引用，避免版本错配。