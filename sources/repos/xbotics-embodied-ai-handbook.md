# Xbotics-Embodied-AI-Handbook（Xbotics 具身智能教程 21 讲）

- **标题**: Xbotics 具身智能教程（21 讲）
- **链接**: https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook
- **类型**: repo / course / hands-on-tutorial
- **作者**: Xbotics-Embodied-AI-club（Xbotics 具身智能社区）
- **许可**: MIT（`LICENSE`，Copyright (c) 2026 Xbotics Embodied AI Club）
- **入库日期**: 2026-10-12（核查对象：`main` 分支，HEAD `aa0e14f`，提交日 2026-10-04）
- **姊妹仓库**: [Xbotics-Embodied-Guide](xbotics-embodied-guide.md)（全局路线 / 公司图谱）；本仓是其"动手跑完整闭环"的配套课程
- **代码：** 同仓 `code/`（见下「开源范围核查」）
- **项目页：** 未发现独立项目页；仓库另有 `docs/github-pages` 分支（入库日未核实是否已部署站点）

## 定位

README 自述：**不是论文综述，也不是纯科普**，而是围绕真实机器人任务、可跑通 Demo、可复盘失败的工程路径；"Guide 帮你看全局，本教程带你动手跑完整闭环"。由原"十八讲"扩展为 21 讲，分六个部分、五个阶段项目。

## 课程结构（`docs/SUMMARY.md`）

| 部分 | 讲次 | 内容 | 阶段项目 |
|------|------|------|----------|
| Part 1 系统基础 | L1–L4 | 导论、ROS2 架构、本体与控制、传感器/坐标系 | 基础控制闭环 |
| Part 2 视觉操作 | L5–L7 | 仿真任务搭建、位姿估计、操作技能与失败恢复 | 视觉定位与模块化抓取 |
| Part 3 端到端 | L8–L13 | Pipeline→Policy、数据闭环、ACT/Diffusion/Flow Matching、VLA 导览、VLA 微调、VLA 前沿 | 端到端操作策略学习 |
| Part 4 强化学习 | L14–L16 | PPO 入门、GRPO 后训练、off-policy 与真机 RL | 与策略/运控衔接 |
| Part 5 VLN | L17–L18 | VLN 任务/基准、方法与评测 | 语言驱动导航评测 |
| Part 6 前沿综合 | L19–L21 | Embodied Agent、世界模型（Fast-WAM）、综合答辩 | 移动+操作综合答辩 |

每讲固定结构：本讲目标 → 核心知识点 → 有硬件版 Demo / 无硬件仿真版 Demo → 实验步骤 → 作业交付 → 常见失败与复盘 → 参考开源项目（`templates/lecture-template.md`）。

## 硬件与仿真双路径（`docs/00-preface/02-dual-path.md`）

| 真机 | 无硬件替代 |
|------|-----------|
| SO101 教学机械臂 | MuJoCo / ManiSkill / robosuite / MoveIt2 |
| xLeRobot 移动操作套件 | TurtleBot3 + 机械臂 mock / ROS2 mock / Isaac Lab |
| Imeta-Y1 科研机械臂（优先级较低） | 通用 6DoF / MoveIt2 / MuJoCo / Isaac Lab |
| Unitree G1 | unitree_rl_gym / unitree_rl_lab / unitree_sim_isaaclab |

## 关键章节摘录（均出自仓内文稿）

1. **L8 端到端策略导论**：训练损失低 ≠ 任务成功；闭环误差会滚雪球，必须 rollout 验证；动作口径（绝对/相对）、归一化统计量与模型绑定、裁剪限位与"出发前检查单"。以 π₀ 在 LIBERO 上闭环部署为演示。
2. **L9 操作数据闭环**：把采集方法分成四类——真机数据（Leader–Follower）、人类代理数据（手持夹爪/手套）、通用接口遥操作（手柄/VR）、无机器人数据（第一视角视频/世界模型）——沿"轨迹可执行性 vs 可规模化"轴权衡；"数据决定上限"归纳为规模律、跨本体正迁移、质量、多样性四类证据。
3. **L12 VLA 微调**：全量 SFT 与 LoRA/QLoRA 对比；`lerobot-train` 全参微调 `lerobot/smolvla_base`（SmolVLM2-500M + flow-matching 动作专家）到 SO-101；仿真+真机混合成 3698 集训练；README 展示三个仿真任务成功率 罐 96% / 小方块 90% / 方块 94%（均过 70% 验收线）。真机验收**不能用相关系数**（`state` 照抄也有 corr 0.936–0.995），改用 `ratio`、`delta_corr`、`delta_std` 三个相对指标；推理与执行分家（边缘板 + x86 GPU，经 gRPC）。真机九任务权重：`Harrysunshine/so101-smolvla-9task`（Hugging Face）。
4. **L15 GRPO 后训练**：在 Qwen2.5-VL-3B（4bit + LoRA）上用"格式奖励 + 正确性奖励"做 RLVR，SuperCLEVR 固定 200 题数数准确率 **0.095 → 0.44**、格式合规率 0.22 → 0.79（300 步 GRPO，单卡约二十分钟）；并与 VLA 后训练（LIBERO 平均成功率 91.0 → 99.1，文稿引用的参考结果）对照。
5. **L16 真机强化学习**：为何真机要 off-policy（replay buffer 复用经验）；DDPG → TD3 → SAC；FastTD3/FastSAC 在大规模并行仿真上的应用，引用 arXiv 2512.01996 称单张 RTX 4090 约 **15 分钟**训出可 sim-to-real 的 G1/T1 行走策略（文稿注明是同 GPU 墙钟对比，不是"PPO 学不会"）。
6. **L19 Embodied Agent**：六点结论——Agent 围绕目标持续行动；Harness 组织并约束 Planning/Memory/Tool Use；LLM 不直出电机指令，而是高层规划、Skill、执行器、Evaluator、安全层分工；失败 Episode 是改进资产。
7. **L20 世界模型**：以 Fast-WAM 官方代码复现为主线，问题是"测试时显式生成未来是否比只在训练期做视频协同建模带来额外控制收益"；对比 Fast-WAM / Joint / IDM 三条路线，release checkpoint 在 LIBERO 上评测。

## 开源范围核查（项目页/源码，入库日 2026-10-12）

- **整体**：文稿、配图、代码、PDF（Part 3/4 部分讲）均在同仓，MIT。
- **代码成熟度不均**（据 `code/` 文件数与 README，**以下为按文件内容的判断**）：
  - 较完整：`code/vla/`（117 文件，数据采集 / ACT / OpenVLA、π₀-FAST、π₀.₅、SmolVLA 推理 / SmolVLA 全量与 LoRA 微调 / SO-101 真机 SFT 部署）、`code/rl/`（87 文件，G1 行走 SAC、动作跟踪、GRPO VLM 数数、CartPole 值函数）、`lecture02`、`lecture05`、`lecture07`。
  - 较薄：`lecture01/03`（2–4 文件）、`lecture19/20`（mock / 简版 demo，`simulation/` 下各 2 个脚本）。
  - 占位：`lecture17/18/21` 仅 README + `hardware/`，README 写的是 Habitat VLN 走读、"Agent + VLN + 操作全链路 mock"，尚无实质实现。
- **进度**：`meta/status.md` 的里程碑/初稿/修改多数仍标 ⬜（仅 L17–L21 文稿标 ✅），与 README 中 21 讲文稿均已存在的状态有出入；以仓内文件为准，状态表可能未同步。
- **外部依赖**：LeRobot、LIBERO、Fast-WAM（`yuantianyuan01/FastWAM`）、Isaac Lab、MuJoCo Menagerie 等链接集中在 `references/links.md`。

## 对 Wiki 的映射
- **wiki/entities/xbotics-embodied-ai-handbook.md**（新建，课程实体页）
- **wiki/entities/lerobot.md**（补充教程配套关联）
- 相关既有页：`wiki/methods/vla.md`、`wiki/methods/imitation-learning.md`、`wiki/methods/reinforcement-learning.md`、`wiki/methods/grpo.md`、`wiki/entities/paper-fast-wam.md`、`wiki/entities/libero-benchmark.md`
