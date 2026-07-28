# TeleOpBench（cyjdlhy/TeleOpBench）

> 来源归档

- **标题：** TeleOpBench
- **类型：** repo / benchmark
- **来源：** 上海人工智能实验室等
- **链接：** <https://github.com/cyjdlhy/TeleOpBench>
- **项目页：** <https://gorgeous2002.github.io/TeleOpBench/>
- **论文：** <https://arxiv.org/abs/2505.12748>
- **许可：** Apache-2.0（仓库同时列出上游组件各自许可）
- **入库日期：** 2026-07-28
- **一句话说明：** TeleOpBench 官方仓库，公开人形/灵巧手模型、四类遥操作接收与重定向模块，以及相机和 Vision Pro 的运行说明。
- **开源状态：** **部分开源** — `teleop/` 可辨识运行入口已公开；论文宣称的 30 场景完整 benchmark 复现实验链未在 README 中闭合。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-teleopbench-a-simulator-centric-benchmark-for-du.md`](../../wiki/entities/paper-notebook-teleopbench-a-simulator-centric-benchmark-for-du.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 环境 | `environment.yml`、`requirements.txt` |
| 资产 | `assets/`（GR1、G1、H1、灵巧手等） |
| 入口 | `teleop/run.sh` / `teleop/main.py` |
| 接口 | `receiver_wrapper/`：camera、Vision Pro、exoskeleton、Xsens |
| 控制 | `robot_control/`：PINK/IK 与 dex-retargeting |
| 缺口 | README 未给 30 任务统一启动、重置、成功判定与结果汇总命令 |

## 对 wiki 的映射

- 项目页：[`teleopbench-project.md`](../sites/teleopbench-project.md)
- 论文来源：[`humanoid_pnb_teleopbench.md`](../papers/humanoid_pnb_teleopbench.md)
