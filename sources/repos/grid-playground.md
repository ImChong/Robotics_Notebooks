# GRID-playground

> 来源归档

- **标题：** GRID Playground
- **类型：** repo
- **链接：** <https://github.com/GenRobo/GRID-playground>
- **历史：** arXiv:2310.00887 脚注曾指向 `ScaledFoundations/GRID-playground`；现由 **GenRobo** 维护
- **入库日期：** 2026-09-27
- **一句话说明：** General Robotics **GRID** 的 **示例 notebook + 仿真 JSON 配置**（AirGen 无人机/车、Isaac 人形/四足/UR5e 等），用于在 GRID 平台会话中跑感知、控制、数据生成与 RL 教程。

## 开源状态

- **已开源（Playground 子集）**：仓库含 `notebooks/`、`configs/airgen/`、`configs/isaac/`、`LICENSE`。
- **非开源：** GRID 平台本体、Cortex 模型权重与 Enterprise 训练栈不在此仓。

## 目录要点

| 路径 | 用途 |
|------|------|
| `notebooks/hello_grid.ipynb` | AirGen 车辆控制 + 分割模型示例 |
| `notebooks/drone_*.ipynb` | 无人机控制、GPS 导航、路径规划、热成像 |
| `notebooks/grid-isaac/*.ipynb` | Isaac 人形 AI、locomotion、Franka/UR  reach |
| `configs/airgen/*.json` | 无人机/车/仓库等多传感器场景 |
| `configs/isaac/*.json` | G1/H1/Go2/Anymal/UR5e 等场景 |

## 对 wiki 的映射

- [paper-grid-general-robot-intelligence-development.md](../../wiki/entities/paper-grid-general-robot-intelligence-development.md)
- [grid-general-robotics.md](../../wiki/entities/grid-general-robotics.md)
