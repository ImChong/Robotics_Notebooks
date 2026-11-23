# EGalahad/sim2real

> 来源归档

- **类型：** repo
- **链接：** https://github.com/EGalahad/sim2real
- **访问日期：** 2026-10-13
- **复核日期：** 2026-11-21
- **核对范围：** 完整阅读 EGalahad/sim2real 官方 README，核查 scripts/tracking_experiment/README.md、run_motiondecode_eval.py、run_tracking_metrics_eval.py、compute_tracking_metrics.py 中的命令、执行顺序、计分常量和输出，以及 manifests/motiondecode.json 固定的数据集 revision；对照 RoboParty Party OS MimicLite 官方指南 §6。
- **通读：** 已完成
- **一句话说明：** 面向 Unitree G1 的模块化 sim2sim/sim2real 策略推理、MuJoCo 跟踪、遥操作与真机 I/O 框架。

## 仓库要点

- Root project 提供策略推理、动作跟踪、MuJoCo sim2sim 和机器人 I/O；独立的 venv/pico 环境负责 Pico/XR retargeting 与动作记录。
- 文档描述两种硬件布局：x86_64 PC 经 Ethernet 控制 G1，或在 G1 机载 Orin 上本地运行整条 pipeline。策略 runtime 与执行 backend 解耦：sim2sim 用 MuJoCo，sim2real 使用不同 Robot I/O 模式。
- README 列出 Mimic Lite v1.1、MimicLite-ROA/PPO、HEFT、HoloMotion、SONIC 家族、GRIT、ScaleBFM、BFM-Zero、TeleopIT、Humanoid-GPT 和 TWIST2 等适配/分发策略。ONNX 文件本身不足以完成移植，仍需对齐观测历史、关节顺序、增益和动作缩放语义。
- 大型 checkpoint 与 third-party 运行时资产不纳入 git，需从项目提供的 artifacts 下载；代码仓库和权重资产的更新节奏应分别核验。
- 快速开始使用 uv 安装 CPU 推理依赖，并通过 MuJoCo 启动 G1 动作跟踪。仓库附有 Codex skills 进行策略适配和 G1 安装/修复。

## 评测资源

排行榜采用冻结的 MotionDecode 子集和共享 MuJoCo evaluator；评估策略使用 CPU ONNX Runtime，保存完整轨迹后离线计算指标。详细流程与指标口径见 [Leaderboard 来源归档](../sites/egalahad-sim2real-leaderboard.md) 与 [Cross-Codebase Evaluation](https://egalahad.github.io/sim2real/tutorials/cross-codebase-evaluation/)。

## 官方榜单代码执行路径（2026-11-21）

[跨代码库跟踪评测文档](https://egalahad.github.io/sim2real/tutorials/cross-codebase-evaluation/) 与 [RoboParty MimicLite 使用指南 §6](https://roboparty.com/documents/party-os?doc=mimiclite&lang=zh) 均指向同一评测栈。源码入口如下：

1. `scripts/tracking_experiment/run_motiondecode_eval.py` 解析 `manifests/motiondecode.json`，按固定 revision 下载四组 G1 MotionDecode 子集，校验 ordered motion list 与每文件 SHA256；默认处理 locomotion、manipulation。
2. `run_tracking_metrics_eval.py` 逐 `policy × motion × seed` 启动 `python -m sim2real.sim_env.integrated_sim2sim --inference-backend onnx-cpu --headless --run-once`；以 `--trajectory-output` 存完整 NPZ。
3. `compute_tracking_metrics.py` 从 NPZ 计算故障前的局部 body/wrist 误差、tracking return、progress 与 root endpoint；批量运行器写 `runs.csv`、`tracking_metrics.csv/json`、`summary.json`，顶层写 `motiondecode_summary.json`。
4. 当前代码中的高度差阈值为 **0.3 m**（投影重力差阈值 0.8）。如见旧版 README 的 0.25 m 或轨迹均值 Root 描述，以当前源代码和 Cross-Codebase 文档为准。

这里记录的是**静态源码与文档核验**，未下载评测权重或运行 MuJoCo 基准。