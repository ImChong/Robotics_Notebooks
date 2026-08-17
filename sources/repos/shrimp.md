# Wisc-HCI/SHRIMP

- **标题：** SHRIMP 官方实现
- **类型：** repo
- **URL：** <https://github.com/Wisc-HCI/SHRIMP>
- **许可：** 仓内无 SPDX LICENSE 文件
- **配套论文：** [arXiv:2608.08884](https://arxiv.org/abs/2608.08884) — [`sources/papers/shrimp_arxiv_2608_08884.md`](../papers/shrimp_arxiv_2608_08884.md)
- **项目页：** <https://wisc-hci.github.io/SHRIMP/>
- **入库日期：** 2026-08-17

## 一句话说明

双机 Docker：COMPUTER 2 跑 Isaac Sim + 界面，COMPUTER 1 跑 Franka FCI；自然语言层级计划在仿真里迭代后再上双臂。

## 仓库状态（2026-08-17 核查）

| 项 | 内容 |
|----|------|
| 启动 | `setup_scripts/start_desktop.sh`、`start_laptop.sh` |
| 子模块 | `libs/robot-stack` → `Wisc-HCI/robot-stack` |
| 标定 | `calibrate_T_world_color.py`（ArUco） |
| 硬件 | 2× Panda 7-DoF + Tesollo Dg-3F + RealSense |

最短复现（需硬件）：`git submodule update --init --recursive` → 双机 Docker → `./setup_scripts/start_desktop.sh`。

## 与 wiki 的关系

- 实体页：[paper-shrimp](../../wiki/entities/paper-shrimp.md) — 含源码运行时序图。
