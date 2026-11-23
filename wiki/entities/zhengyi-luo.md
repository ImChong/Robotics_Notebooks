---
type: entity
tags: [humanoid, teleoperation, sim2real, motion-tracking, egocentric-vision, cmu, nvidia, gear]
status: complete
updated: 2026-11-22
related:
  - ./agentsdock.md
  - ./smplolympics.md
  - ./jim-fan.md
  - ./tairan-he.md
  - ./nvidia-sonic.md
  - ../tasks/teleoperation.md
  - ../concepts/motion-retargeting.md
  - ../concepts/sim2real.md
  - ./mimickit.md
  - ./humanoid-robot.md
  - ./nvidia-gear-lab.md
sources:
  - ../../sources/sites/zhengyiluo.md
original:
  - ../../sources/sites/zhengyiluo.md
summary: "罗正宜（Zhengyi Luo）为 NVIDIA GEAR Lab 高级研究科学家、CMU RI 博士（Kris Kitani）；工作横跨人形通用低层控制、人–人形遥操作、视觉 Sim2Real 与交互感知，是 HOVER、ASAP、OmniH2O、PDC 与 SONIC 等社区关键论文的核心作者之一。"
---

# Zhengyi Luo（罗正宜）

## 一句话定义

**Zhengyi Luo** 的研究把 **人形机器人的通用低层控制** 与 **视觉–语言–动作、Sim2Real 与遥操作数据闭环** 串在同一职业轨迹上：博士阶段提出并开源 **PHC / PULSE** 等「可扩展人形控制与表示」路线，随后在 NVIDIA GEAR 与 LECAR 网络合作者一道推进 **HOVER、ASAP、VIRAL、Doorman、SONIC** 等面向真机与大规模仿真的系统论文。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |

## 为什么重要

- **人形学习管线枢纽作者**：与 [Tairan He](./tairan-he.md) 等在 **OmniH2O、HOVER、ASAP、SONIC、VIRAL** 上高度重叠署名，是理解 **2023–2026 通用人形控制 + 视觉迁移** 论文簇的关键人物节点；所属研究组见 [NVIDIA GEAR Lab](./nvidia-gear-lab.md)（PI：[Jim Fan](./jim-fan.md)、Yuke Zhu）。
- **开源资产多**：PHC、PULSE、SimXR、EmbodiedPose、UniversalHumanoidControl 等仓库长期被社区用作 **仿真人形 baseline 与重定向参考**（以各仓库 README 为准）。
- **编码代理基础设施**：[AgentsDock](./agentsdock.md)（跨端 IDE + 自托管 AgentsServer）面向 agentic AI 研究长跑场景，与上述人形论文栈正交但同属其 GitHub 生态。

## 核心研究脉络（归纳）

1. **通用控制与表示**：论文与代码强调在单一策略或表示下覆盖多种步态 / 技能（博士论文题目 *Learning Universal Humanoid Control* 概括该主线）。
2. **人–机共享演示**：OmniH2O、实时全身遥操作等把 **接口设计、数据规模与策略学习** 绑在一起，对应 [Teleoperation](../tasks/teleoperation.md) 任务页讨论的问题域。
3. **视觉与 Sim2Real**：PDC、VIRAL、Doorman 等工作把 **像素观测、规模化仿真与真机迁移** 推到前台，与 [Sim2Real](../concepts/sim2real.md) 概念页互补。
4. **与合作者方法的交集**：例如 [CLoSD](https://guytevet.github.io/CLoSD-page/)、[MaskedManipulator](https://xbpeng.github.io/projects/MaskedManipulator/index.html) 体现与 **物理角色控制 / MimicKit 生态** 的交叉。

## 核心原理：研究方法主线（跨论文归纳）

以下是对作者官方 [Publications](https://www.zhengyiluo.com/publications/) 所列研究的**主题归纳**，不代表作者在一次演讲中逐字提出的统一算法：

1. **动作表示与物理控制（PHC / PULSE）**：以动作先验、物理模拟和通用人形控制为研究起点，解决动作保真与动力学可行性之间的矛盾。
2. **人到人形的示范桥接（OmniH2O / Human2Humanoid）**：研究全身遥操作和从人类动作迁移机器人，为数据采集与模仿学习提供动作接口。
3. **从控制到部署（HOVER / ASAP / SONIC）**：研究全身策略的技能覆盖、仿真-真机动力学对齐与规模化运动追踪，工程实现分别以对应官方论文和代码为准。
4. **从单纯动作到视觉交互（VIRAL / GRAIL）**：进一步将视觉感知、物体交互和世界模型生成的数据用于人形 loco-manipulation。

## 工程实践：如何定位一手复现入口

| 需求 | 官方入口 | 核验边界 |
|---|---|---|
| 查履历和研究方向 | [个人学术主页](https://www.zhengyiluo.com/) | 以作者自述为准，部分出版状态会更新 |
| 查论文与项目对应关系 | [官方出版物清单](https://www.zhengyiluo.com/publications/) | 检查论文、项目及 Code 链接是否分别列出 |
| 找作者公开代码 | [GitHub 账号](https://github.com/ZhengyiLuo) | 需核对具体仓库及许可 |
| 复现 SONIC 控制模块 | [SONIC 官方代码](https://github.com/NVlabs/GR00T-WholeBodyControl) | 使用论文版本及仓库文档检查支持的机器人、权重和部署要求 |

原始站点的全文核对范围和日期保存在 [作者主页一手归档](../../sources/sites/zhengyiluo.md)；**这不是 2026-11-18 演讲的正式逐字稿**。

## 局限与风险

- **「Universal」有边界**：通用控制在分布外地形、外骨骼负载或新硬件上仍需 **再训练或对齐模块**（参见 ASAP 等论文主张，以原文为准）。
- **in submission / workshop 标注**：主页出版物区部分条目状态会更新，引用前请核对 **最新会议或 arXiv 版本**。
- **不同证据级别**：作者主页可以验证职位、署名与论文链接；算法性能、实验数字需回到相应论文，GRASP 演讲的口头论断也不能由个人主页替代核实。

## 2026-11-18 GRASP 演讲：从运动追踪到 WAM

演讲原始活动信息及一手材料核对见 [SONIC × GR00T 2 Turbo 双层闭环编译](../queries/sonic-gr00t2-wam-humanoid-dual-brain.md)。本文把 SONIC 的全身跟踪、GRAIL 的合成操作数据与 GR00T 2 Turbo 的板载 WAM 连接起来，区分经白皮书验证的指标和中文转述中的研究观点。

## 关联页面

- [Jim Fan（范林熹）](./jim-fan.md)
- [Tairan He（何泰然）](./tairan-he.md)
- [SONIC（规模化运动跟踪）](./nvidia-sonic.md)
- [NVIDIA GEAR Lab](./nvidia-gear-lab.md)
- [Teleoperation](../tasks/teleoperation.md)
- [Motion Retargeting](../concepts/motion-retargeting.md)
- [Sim2Real](../concepts/sim2real.md)
- [MimicKit](./mimickit.md)
- [人形机器人](./humanoid-robot.md)
- [SMPLOlympics](./smplolympics.md) — 博士阶段主导的 SMPL 仿真人形体育 benchmark
- [AgentsDock](./agentsdock.md) — 自托管编码代理 IDE（Claude Code / Codex / Cursor 等）

## 参考来源

- [Zhengyi Luo 个人主页原始资料](../../sources/sites/zhengyiluo.md)

## 推荐继续阅读

- [机器人论文阅读笔记：PULSE](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/01_Foundational_RL/PULSE_Physics-based_Universal_Latent_Space/PULSE_Physics-based_Universal_Latent_Space.html)
- [HOVER 项目页](https://hover-versatile-humanoid.github.io/)
- [ASAP 项目页](https://agile.human2humanoid.com/)
