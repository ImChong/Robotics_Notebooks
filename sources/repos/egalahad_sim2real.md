# EGalahad/sim2real

> 来源归档

- **标题：** sim2real: A Lightweight and Modular Sim2sim and Sim2real Deployment Stack
- **类型：** repo（仿真、动作跟踪评测与真机部署）
- **所有者：** RoboParty Lab / EGalahad
- **链接：** <https://github.com/EGalahad/sim2real>
- **文档：** <https://egalahad.github.io/sim2real/>
- **Leaderboard：** <https://egalahad.github.io/sim2real/leaderboard>
- **许可证：** 以仓库 LICENSE 为准
- **开源状态：** 代码已公开；大型 checkpoints 与 third_party 运行时资产由共享云盘提供
- **核查日期：** 2026-10-13
- **一句话说明：** 轻量模块化的人形机器人 sim2sim / sim2real 运行栈，提供 MuJoCo 仿真、运动跟踪策略接口、PICO 遥操作和 Unitree G1 I/O，并附跨代码库动作跟踪评测。

## README 核心摘录

- 通用依赖通过 uv sync --extra inference-cpu 安装；G1 真机 I/O 使用独立的 robot extra。
- 离线流程由 MuJoCo 仿真进程和策略跟踪进程组成；两进程通过 bridge 交互。
- 仓库将策略 observation/action 约定、MuJoCo bridge、遥操作和机器人 I/O 拆开，供外部策略接入同一运行时。
- 大型运行时文件不在 Git 中，需按 README 从共享云盘下载 checkpoints 与 third_party 资产；复现时要记录所用配置和权重版本。
- Motion Tracking Leaderboard 对策略使用同一组 held-out 动作、seed 和终止条件，但保留各策略原生 observation / action processing；阅读数字时应区分评测协议统一与模型实现相同。

## 对 wiki 的映射

- [sim2real 项目与排行榜](../../wiki/entities/sim2real.md)
- [文档站与排行榜来源](../sites/egalahad-sim2real.md)
- [Sim2Real 概念](../../wiki/concepts/sim2real.md)
