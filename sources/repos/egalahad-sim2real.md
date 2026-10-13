# EGalahad/sim2real

> 来源归档

- **类型：** repo
- **链接：** https://github.com/EGalahad/sim2real
- **访问日期：** 2026-10-13
- **一句话说明：** 面向 Unitree G1 的模块化 sim2sim/sim2real 策略推理、MuJoCo 跟踪、遥操作与真机 I/O 框架。

## 仓库要点

- Root project 提供策略推理、动作跟踪、MuJoCo sim2sim 和机器人 I/O；独立的 venv/pico 环境负责 Pico/XR retargeting 与动作记录。
- 文档描述两种硬件布局：x86_64 PC 经 Ethernet 控制 G1，或在 G1 机载 Orin 上本地运行整条 pipeline。策略 runtime 与执行 backend 解耦：sim2sim 用 MuJoCo，sim2real 使用不同 Robot I/O 模式。
- README 列出 Mimic Lite v1.1、MimicLite-ROA/PPO、HEFT、HoloMotion、SONIC 家族、GRIT、ScaleBFM、BFM-Zero、TeleopIT、Humanoid-GPT 和 TWIST2 等适配/分发策略。ONNX 文件本身不足以完成移植，仍需对齐观测历史、关节顺序、增益和动作缩放语义。
- 大型 checkpoint 与 third-party 运行时资产不纳入 git，需从项目提供的 artifacts 下载；代码仓库和权重资产的更新节奏应分别核验。
- 快速开始使用 uv 安装 CPU 推理依赖，并通过 MuJoCo 启动 G1 动作跟踪。仓库附有 Codex skills 进行策略适配和 G1 安装/修复。

## 评测资源

排行榜采用冻结的 MotionDecode 子集和共享 MuJoCo evaluator；评估策略使用 CPU ONNX Runtime，保存完整轨迹后离线计算指标。详细流程与指标口径见 [Leaderboard 来源归档](../sites/egalahad-sim2real-leaderboard.md) 与 [Cross-Codebase Evaluation](https://egalahad.github.io/sim2real/tutorials/cross-codebase-evaluation/)。
