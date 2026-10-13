---
type: entity
tags: [sim2real, humanoid, motion-tracking, sim2sim, deployment, evaluation, unitree-g1, roboparty]
status: complete
updated: 2026-10-13
project_id: egalahad-sim2real-stack
project: https://egalahad.github.io/sim2real/
code: https://github.com/EGalahad/sim2real
summary: "EGalahad sim2real 是用于人形策略仿真验证、外部策略适配和真机部署的模块化运行栈，并提供 Unitree G1 全身动作跟踪的跨代码库排行榜。"
related:
  - ../concepts/sim2real.md
  - ./mimiclite.md
  - ./paper-heft.md
  - ./holomotion.md
  - ../methods/sonic-motion-tracking.md
  - ./grit.md
  - ./paper-scaling-bfm-humanoid.md
  - ./paper-bfm-zero.md
  - ./paper-teleopit.md
  - ./paper-humanoid-gpt.md
  - ./paper-twist2.md
  - ./grit-deploy-onboard-g1.md
sources:
  - ../../sources/sites/egalahad-sim2real.md
  - ../../sources/repos/egalahad_sim2real.md
---

# EGalahad sim2real：运行栈与动作跟踪排行榜

**一句话定义：** [sim2real](https://github.com/EGalahad/sim2real) 是一套面向 Unitree G1 的模块化 sim2sim / sim2real 运行栈，支持外部全身跟踪策略适配、MuJoCo 评测、PICO 遥操作与机器人 I/O；文档站附有跨代码库匹配评测排行榜。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Sim | Simulation to Simulation | 在仿真器之间验证策略适配与行为一致性 |
| Sim2Real | Simulation to Real | 将仿真策略迁移并部署到真实机器人 |
| WBT | Whole-Body Tracking | 按参考动作驱动人形机器人全身运动 |
| SDK | Software Development Kit | G1 真机接口所需的软件开发套件 |
| ONNX | Open Neural Network Exchange | 仓库支持的策略模型交换 / 推理格式 |

## 为什么重要

- **连接训练代码与运行时：** 将策略 observation/action 约定、MuJoCo bridge、遥操作和机器人 I/O 拆开，便于对照训练端实现和部署端数据契约。
- **可复用的横向评测：** 同一评测器按统一参考动作和终止规则比较多种开源 / 发布策略，减轻复现者自行拼装异构测试环境的工作。
- **把“榜单排名”放回协议中读：** 分值受身体误差、根轨迹、手腕误差、完成度和跟踪回报共同影响；不同策略仍使用自身 observation 与动作处理栈，不能把榜单理解为算法间完全同条件的训练竞赛。

## 运行架构

| 边界 | 主要模块 | 作用 |
|------|----------|------|
| 策略适配 | 外部策略 observation / action 接口、YAML 配置 | 保持各策略输入顺序、时序和推理契约 |
| 仿真端 | MuJoCo simulator、UDP bridge、mjviser | 运行 G1 模型、回放动作并观察 sim2sim 结果 |
| 策略端 | tracking / inference process | 接收参考运动与机器人状态，推理关节目标 |
| 真机端 | Unitree SDK2 / robot I/O | 将策略输出送往机器人控制接口 |
| 遥操作 | PICO、XRoboToolkit、retargeting service | 将操作者动作转为机器人参考运动 |

部署或复现时，先确认策略配置、权重、动作参考帧率、关节次序、观察量归一化和通信时序一致，再从 MuJoCo 测试逐步进入受保护的真机验证。

## Leaderboard 协议

官方 [Motion Tracking Leaderboard](https://egalahad.github.io/sim2real/leaderboard/) 将策略在 held-out MotionDecode 数据上比较，默认集合为 Locomotion-80、Manipulation-48、Ground-60 和 Dance-40；Legacy LAFAN-40、PHUMA-30、Root-90 可选。指标包括身体局部位置和姿态误差、全局根误差、训练 GPU 小时、左右手腕误差、Tracking Return 与 Progress。

协议固定 seed 0、无运行时上限，并统一参考动作和失败终止条件。各策略依然保留原生 observation 与 action processing，所以榜单比较的是共享评估协议下的部署策略，不是训练成本、架构或训练数据都相同的消融实验。GPU Hours 仅在找到可辩护的官方来源时呈现；缺失表示没有可靠来源，不应自行补估。

## 榜单条目与项目详情映射

15 个策略条目对应 10 个项目家族。同一项目的模型规模、训练版本或推理配置映射到同一主详情页；每个项目详情页只解释一个项目身份。

| 排行榜条目 | 项目详情节点 | 身份关系 |
|---|---|---|
| Mimic Lite v1.1 | [MimicLite](./mimiclite.md) | 同一训练与部署基础设施的版本 |
| MimicLite-ROA | [MimicLite](./mimiclite.md) | 同一项目的 ROA checkpoint |
| MimicLite-PPO | [MimicLite](./mimiclite.md) | 同一项目的 PPO checkpoint |
| HEFT | [HEFT](./paper-heft.md) | 独立论文 / 方法项目 |
| HoloMotion | [HoloMotion](./holomotion.md) | 独立项目与官方实现 |
| SONIC | [SONIC](../methods/sonic-motion-tracking.md) | 共享 SONIC 主项目节点 |
| SONIC-Low-Latency | [SONIC](../methods/sonic-motion-tracking.md) | 同一策略家族的低延迟变体 |
| SONIC-v1.1 | [SONIC](../methods/sonic-motion-tracking.md) | 同一策略家族的版本 |
| GRIT v0.0.1 | [GRIT](./grit.md) | 独立策略 / 部署仓库 |
| ScaleBFM-XL | [ScaleBFM](./paper-scaling-bfm-humanoid.md) | 同一论文 / 项目的 XL 模型 |
| ScaleBFM-M | [ScaleBFM](./paper-scaling-bfm-humanoid.md) | 同一论文 / 项目的 M 模型 |
| BFM-Zero | [BFM-Zero](./paper-bfm-zero.md) | 独立论文 / 策略项目 |
| TeleopIt | [TeleopIt](./paper-teleopit.md) | 独立论文 / 系统项目 |
| Humanoid-GPT | [Humanoid-GPT](./paper-humanoid-gpt.md) | 独立论文 / 通用跟踪器 |
| TWIST2 | [TWIST2](./paper-twist2.md) | 独立论文 / 遥操作系统 |

## 榜单的工程读法

- 误差类指标越低越好；Tracking Return 与 Progress 越高越好；不同指标各自排序，GPU Hours 使用对数轴。
- leaderboard 表格默认展示跨可用数据集汇总值，并在次行列出 Locomotion / Manipulation 子集；Global Root 在 MotionDecode 仅对 Locomotion 提供。
- 论文报告、排行榜复现实验与部署策略版本可能不同。需要对比论文数字时，先核对参考动作集、坐标对齐、终止条件和统计口径；不可将 leaderboard 数值移植成原论文结论。
- 缺失 GPU 时长表示没有可验证来源，并不代表训练成本较低或为零。

## 开源状态与复现边界

- **已开源：** 仓库提供仿真、策略适配、遥操作与真机运行时代码。
- **外置资产：** 大型 checkpoint 与 third_party 文件从共享云盘获取，不在 Git 仓库追踪。
- **部分项目实现：** GRIT 仓库发布推理、策略 ONNX 与部署工具链，但 README 明确将训练 / 数据生成排除在仓库范围外；其版本计划须再核对当前仓库发布。
- **复现边界：** 固定评测协议减少了 evaluator 差异，但不同 policy 仍保留自身 observation / action processing，且 GPU 小时只从官方可核查来源收录。

## 关联页面

- [Sim2Real 工程概念](../concepts/sim2real.md)
- [GRIT v0.0.1 策略仓库](./grit.md)
- [GRIT 机载集成工程](./grit-deploy-onboard-g1.md)
- [MimicLite](./mimiclite.md)
- [SONIC](../methods/sonic-motion-tracking.md)
- [ScaleBFM](./paper-scaling-bfm-humanoid.md)
- [BFM-Zero](./paper-bfm-zero.md)
- [HoloMotion](./holomotion.md)
- [HEFT](./paper-heft.md)
- [TeleopIt](./paper-teleopit.md)
- [Humanoid-GPT](./paper-humanoid-gpt.md)
- [TWIST2](./paper-twist2.md)

## 参考来源

- [sim2real 文档与排行榜来源](../../sources/sites/egalahad-sim2real.md)
- [sim2real GitHub 仓库归档](../../sources/repos/egalahad_sim2real.md)
- [GRIT v0.0.1 代码归档](../../sources/repos/grit_teleop_deploy.md)

## 推荐继续阅读

- [项目文档](https://egalahad.github.io/sim2real/)
- [Motion Tracking Leaderboard](https://egalahad.github.io/sim2real/leaderboard/)
- [Cross-Codebase Tracking Evaluation](https://egalahad.github.io/sim2real/tutorials/cross-codebase-evaluation/)
- [EGalahad/sim2real GitHub](https://github.com/EGalahad/sim2real)
