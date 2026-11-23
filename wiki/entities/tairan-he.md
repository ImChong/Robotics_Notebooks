---
type: entity
tags: [humanoid, sim2real, loco-manipulation, teleoperation, visual-rl, reinforcement-learning, cmu, nvidia, openai]
status: complete
updated: 2026-11-22
related:
  - ./paper-viral-humanoid-visual-sim2real.md
  - ./paper-doorman-opening-sim2real-door.md
  - ./gr00t-visual-sim2real.md
  - ./humanoid-robot.md
  - ./jim-fan.md
  - ./zhengyi-luo.md
  - ./xue-bin-peng.md
  - ./yanjie-ze.md
  - ../concepts/sim2real.md
  - ../tasks/loco-manipulation.md
  - ../tasks/teleoperation.md
  - ./nvidia-sonic.md
  - ../concepts/privileged-training.md
sources:
  - ../../sources/sites/tairan-he.md
original:
  - ../../sources/sites/tairan-he.md
summary: "何泰然（Tairan He）现为 OpenAI Member of Technical Staff；CMU RI 博士（2026-04 答辩）、NVIDIA GEAR 实习背景，研究聚焦人形规模化学习与视觉 Sim2Real；代表作含 OmniH2O、HOVER、ASAP、VIRAL / DoorMan 等。"
---

# Tairan He（何泰然）

## 一句话定义

**Tairan He** 是面向 **通用人形 loco-manipulation** 的机器学习研究者：2026 年起任 **OpenAI Member of Technical Staff**（*building physical AGI*）；博士阶段在 CMU LECAR 与 NVIDIA GEAR 合作网络中系统推进 **特权教师 RL → 视觉学生 / 蒸馏 / 动力学对齐** 的 Sim2Real 管线，并以 [OmniH2O](https://omni.human2humanoid.com/)、[VIRAL](https://viral-humanoid.github.io/) 等工作把 **全身遥操作接口与视觉迁移** 推到社区可见的基准线。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| MTS | Member of Technical Staff | OpenAI 等技术机构的研究/工程职级称谓 |
| RGB | Red-Green-Blue | 彩色图像通道，常与深度 (RGB-D) 配合 |

## 为什么重要

- **问题域与仓库主线重合**：其公开论文串覆盖 [Sim2Real](../concepts/sim2real.md)、[Privileged Training](../concepts/privileged-training.md)、[Loco-Manipulation](../tasks/loco-manipulation.md)、[Teleoperation](../tasks/teleoperation.md) 与规模化仿真训练，是理解「2023–2026 人形学习论文潮」的 **作者级索引**。
- **职业轨迹具风向标意义**：从 CMU LECAR / NVIDIA GEAR 博士训练到 **OpenAI 通用人形方向**（主页 2026-05 新闻），与 [Zhengyi Luo](./zhengyi-luo.md)（NVIDIA GEAR）、[Yanjie Ze](./yanjie-ze.md)（Figure AI）等同期人形研究者形成可对照的 **产业–学术迁移谱系**。
- **工程可检索**：个人页集中给出项目站与 BibTeX，比零散抓 arXiv 更适合作为 ingest 溯源与后续 curator 补链入口。

## 职业与研究脉络（归纳）

1. **SJTU 本科 + MSRA**：计算机本科（导师 Weinan Zhang），奠定大规模机器学习基础。
2. **CMU RI 博士 + NVIDIA GEAR 实习**：与 Guanya Shi、Changliu Liu 及 GEAR（[Jim Fan](./jim-fan.md)、Yuke Zhu）合作，产出 OmniH2O、HOVER、ASAP、VIRAL、DoorMan 等 **人形 Sim2Real 主线**；**2026-04 答辩**。
3. **OpenAI（2026-05 起）**：主页首段自述为 *Member of Technical Staff at OpenAI building physical AGI*，新闻栏表述为 *general-purpose robots*。

## 核心研究脉络（技术向）

1. **接口与数据**：OmniH2O 一类工作强调 **人–机共享运动表示** + 遥操作数据，对应「如何低成本获得全身演示」。
2. **仿真规模化 + 视觉迁移**：VIRAL、DoorMan 等强调 **大规模并行仿真与 RGB 策略**（CVPR 2026），与仓库中 [GR00T-VisualSim2Real](./gr00t-visual-sim2real.md) 所归纳的 Teacher–Student 叙事同族。
3. **动力学与多智能体分解**：ASAP（残差 / 对齐）、SoFTA、FALCON 等体现 **把全身问题拆成可学子模块** 或 **显式补偿 sim–real gap** 的路线。

## 流程总览（反复出现的 Sim2Real 骨架）

下列图只描述其多篇论文共享的 **逻辑骨架**，细节奖励、观测与算法名以各论文为准。

```mermaid
flowchart LR
  Sim["规模化仿真\n域随机 / 渲染"] --> Teacher["特权教师\n状态观测 RL"]
  Teacher --> Student["学生策略\nRGB / 稀疏传感"]
  Student --> Align["对齐模块\n蒸馏 / DAgger /\n残差动力学"]
  Align --> Real["真机部署\n零样本或少微调"]
```

## 工程实践（从作者索引进入可核验的研究实现）

本页是**研究者与论文路线索引**，不是可运行的统一控制算法。以下条目从作者官网各论文的自述摘要归纳；训练参数、代码入口和实验协议应以各项目论文/官方仓库为准。

| 工程问题 | 首选入口 | 官网所述技术链条 | 建议核对 |
|---|---|---|---|
| 全身遥操作与演示数据 | [OmniH2O](https://omni.human2humanoid.com/) | 运动学姿态共享接口；人类动作重定向及增强；特权教师模仿；稀疏传感输入部署 | 机器人与操作者映射、重定向误差、传感器边界 |
| RGB 视觉控制连续 loco-manipulation | [VIRAL](https://viral-humanoid.github.io/) | 特权 RL Teacher、delta 动作与 reference state 初始化；RGB Student 使用在线 DAgger + 行为克隆蒸馏 | 教师/学生观测差异、视觉随机化、相机和灵巧手 real-to-sim 对齐 |
| 动作跟踪的动力学 Sim2Real | [ASAP](https://agile.human2humanoid.com/) | 仿真预训练 motion tracking；真机轨迹训练 delta action 模型；将模型反馈至仿真微调策略 | 真机采样与训练安全边界、残差动作定义、目标机器人动力学差异 |
| 多模式全身控制 | [HOVER](https://hover-versatile-humanoid.github.io/) | 多种控制模式在全身运动模仿接口下统一蒸馏 | 各模式的命令空间、状态切换与目标硬件适配 |

**核验路径：** 从本页确认研究问题 → 到各项目页确认论文、代码和数据开放状态 → 对照论文中的观测/动作空间与训练、部署实验 → 再选择目标机器人验证。不能把不同项目的训练模块拼成一个未经论文支持的标准流水线。

## 局限与风险

- **主页 ≠ 最新事实源**：OpenAI 具体项目分工、在投论文状态以 **机构新闻与论文页** 为准；本页不替代一手引用。
- **人物页不是单一实现**：VIRAL 的视觉 Teacher–Student、ASAP 的残差动力学对齐与 OmniH2O 的全身遥操作是不同研究任务，其训练流程不应混同。
- **合著与一作分工**：如 SONIC 等以合作者一作为主叙事站点，阅读时应区分 **个人贡献边界**。
- **「博士阶段」表述已过时**：截至 2026-06 主页，其 **当前任职已为 OpenAI**；引用职业身份时应以最新主页为准。

## 关联页面

- [VIRAL 论文实体](./paper-viral-humanoid-visual-sim2real.md)
- [DoorMan 论文实体](./paper-doorman-opening-sim2real-door.md)
- [GR00T-VisualSim2Real](./gr00t-visual-sim2real.md)
- [人形机器人](./humanoid-robot.md)
- [Sim2Real](../concepts/sim2real.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [Teleoperation](../tasks/teleoperation.md)
- [SONIC（规模化运动跟踪）](./nvidia-sonic.md)
- [Jim Fan（范林熹）](./jim-fan.md)
- [NVIDIA GEAR Lab](./nvidia-gear-lab.md)
- [Zhengyi Luo（罗正宜）](./zhengyi-luo.md)
- [Xue Bin Peng（彭学斌）](./xue-bin-peng.md)
- [Yanjie Ze（迮炎杰）](./yanjie-ze.md)

## 参考来源

- [Tairan He 个人主页原始资料（含 2026-11-22 核对范围）](../../sources/sites/tairan-he.md)

## 推荐继续阅读

- [VIRAL 项目页](https://viral-humanoid.github.io/)
- [OmniH2O 项目页](https://omni.human2humanoid.com/)
- [个人主页](https://tairanhe.com/)
