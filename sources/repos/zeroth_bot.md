# zeroth-robotics/zeroth-bot 官方代码仓库核对归档

> 来源归档（GitHub 源码结构核对）

- **仓库：** <https://github.com/zeroth-robotics/zeroth-bot>
- **类型：** repo
- **许可证：** MIT（根目录 `LICENSE`，Copyright (c) 2024 Jingxiang Mo）
- **核对范围：** 核对 `README.md`、`LICENSE`、`.gitmodules` 及工作区目录结构（检查 `kos-zbot/`、`ksim-gym-zbot/`、`public/`）。
- **通读：** 已完成
- **入库日期：** 2026-11-14

## 仓库核对实况

1. **顶层伞仓（Umbrella Repo）文件**：根目录仅包含 `README.md`、`LICENSE`、`public/wave.webp`、`.gitignore` 与 `.gitmodules`，面向社区汇总项目说明、Discord 入口与构建文档链接（<https://docs.kscale.dev/docs/zeroth-01>）。
2. **Git 子模块挂载关系（`.gitmodules`）**：
   - `kos-zbot` -> <https://github.com/kscalelabs/kos-zbot>：机器人操作系统与硬件抽象层（Feetech 舵机控制、IMU 接口、执行器管理、`kos_zbot` Python API、实时低延迟通信、标定与诊断 CLI）。
   - `ksim-gym-zbot` -> <https://github.com/kscalelabs/ksim-gym-zbot>：GPU 加速强化学习训练流水线（从行走到人体动作模仿的全身控制、SysID 标定执行器、MJCF/URDF 预载模型，训练策略直接经 KOS 导出到真机）。
3. **子模块未初始化边界**：在克隆官方主仓 `zeroth-robotics/zeroth-bot` 时，`kos-zbot/` 与 `ksim-gym-zbot/` 均为空目录（子模块源码未随主仓浅克隆拉取），因此本轮核对无法审阅底层 Python/Rust 驱动及 RL 训练循环源码细节。

## 对 wiki 的映射

- [paper-notebook-zeroth-bot-https-github-com-zeroth-robotics-zero](../../wiki/entities/paper-notebook-zeroth-bot-https-github-com-zeroth-robotics-zero.md)
