---
type: entity
tags: [paper, humanoid, loco-manipulation, latent-planning, flow-matching, unitree-g1, amazon-far, cmu, stanford]
institutions: [amazon-far, cmu, stanford]
status: complete
updated: 2026-10-21
project_id: fiber-humanoid-loco-manipulation
arxiv: "2610.24675"
project: "https://fiberaa.github.io/"
code: "https://github.com/Yutang-Lin/FIBER"
related:
  - ../tasks/loco-manipulation.md
  - ./unitree-g1.md
  - ./paper-motionwam-humanoid-loco-manipulation-wam.md
sources:
  - ../../sources/papers/fiber_arxiv_2610_24675.md
  - ../../sources/sites/fiber.md
  - ../../sources/repos/fiber.md
summary: "FIBER 将异构全身技能专家投影到冻结 BFM 的动作对齐潜空间，再离线训练统一感知 flow planner；G1 仿真平均成功率 42.0%，官方仓库已公开但可运行代码仍待发布。"
---

# FIBER：动作对齐潜空间规划

**FIBER**（*Action-Aligned Latent Planning for Versatile Humanoid Loco-Manipulation*，[arXiv:2610.24675](https://arxiv.org/abs/2610.24675)，[项目页](https://fiberaa.github.io/)）由 **Amazon Frontier AI & Robotics（Amazon FAR）、卡内基梅隆大学（CMU）与斯坦福大学**提出。它把冻结行为基础模型（BFM）可执行的潜变量作为技能接口，让低频规划器决定“调用哪段行为”，同时由 BFM 保留高频状态反馈。

## 一句话定义

FIBER 将多个独立训练的移动操作专家映射到一个冻结 BFM 的动作对齐潜空间，再用单一感知规划器离线学习跨任务的潜变量计划。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 将潜变量命令转为全身动作并持续闭环反馈的冻结行为模型 |
| LPP | Latent Policy Projection | 将任务专家投影为能复现其局部控制响应的 BFM 潜命令 |
| AH-BFM | Adaptive-Horizon Behavior Foundation Model | 支持不同时间跨度的行为基础模型 |
| FB | Forward-Backward | 训练 AH-BFM 的多折扣目标学习方法 |
| G1 | Unitree G1 | 论文的真机人形平台 |

## 为什么重要

高层规划接口若只给末端位姿或接触目标，无法直接表达技能特有的控制响应；若直接规划关节级动作，规划器又要自行承担高频反馈。FIBER 将任务专家的行为压缩为 BFM 可执行的潜变量，使规划动作定义在“它会让机器人做什么”上，而不是仅定义几何目标。

## 核心方法

### 流程总览

```mermaid
flowchart LR
  experts["独立训练任务专家"] --> lpp["LPP：动作对齐投影"]
  bfm["冻结 AH-BFM"] --> lpp
  lpp --> rollouts["闭环潜轨迹"]
  rollouts --> planner["离线感知 flow planner"]
  planner --> latents["潜变量命令"]
  latents --> bfm
  bfm --> robot["Unitree G1 全身执行"]
```

1. **分技能训练专家：** 箱子拾取、开门和抓取分别训练，各自使用任务专属观测与奖励。
2. **动作对齐投影：** 沿专家轨迹求取一段潜变量，让冻结 BFM 在相同访问状态下产生的动作尽量匹配专家响应；这样潜变量继承可执行行为，而不只是抽象目标。
3. **离线训练统一规划器：** 聚合投影后的闭环潜轨迹，使用 flow matching 训练一个感知规划器，条件包括深度、本体感知和任务命令；采集完 rollouts 后不再需要原任务专家参与规划。
4. **自适应时间跨度：** AH-BFM 通过 multi-gamma Forward-Backward 学习兼顾精确控制和较长时域全身行为。
5. **部署分频：** 规划器约 **5 Hz** 重规划潜变量，冻结 BFM 约 **50 Hz** 读取新状态并产生全身动作。

## 实验与评测

| 证据 | 结果 | 读法 |
|------|------:|------|
| 仿真：箱子拾取、开门、物体抓取；每项 128 种变化 | FIBER 平均成功率 **42.0%**（拾箱 42.1%、开门 53.5%、抓取 30.3%） | 跨任务统一规划的主结果 |
| Projected latent replay | **11.6%** | 使用投影专家潜轨迹的回放基线 |
| Kinematic planning | **3.5%** | 规划运动学目标的对照 |
| Unitree G1 真机（每任务 10 次） | 拾箱 6/10、开门 4/10、抓取 2/10，平均 **40%** | 小样本真机结果，与仿真口径不同 |
| 技能串联与 waypoint 引导（定性） | 拾箱后按 waypoint 搬运穿过门口，被绊后可恢复 | 无成功率；**不是独立导航实验** |

## 源码运行时序图

**不适用** — 截至 **2026-10-21**，官方仓库 README 标注代码 coming soon，仓库尚无可运行的训练、投影或部署入口，因此无法据源码核验运行时调用顺序。项目与开源状态见 [官方仓库归档](../../sources/repos/fiber.md)。

## 工程实践与复现边界

- 项目页展示专家、latent projector 和 planner 均在仿真训练，真机演示使用机载深度观测。
- 结果应区分仿真汇总成功率（每任务 128 种变化）与真机小样本结果（每任务 10 次）；项目页展示的 18 段 rollout 是演示集合，不作统计口径。
- 「可控导航」在论文中仅指 waypoint steering 的定性演示，没有目标点到达或路径跟随的量化评测；引用时不要外推为导航能力已验证。
- 官方 GitHub 仓库已公开，许可证为 Apache-2.0；但仓库 README 写明代码 coming soon，现有文件不足以复现训练与部署。
- 当前可复用的工程结论是接口设计与分频思路；环境配置、模型结构细节、训练超参和真机驱动仍需等待代码发布。

## 与其他工作对比

| 路线 | 规划器输出 | 与 FIBER 的差异 |
|------|------------|-----------------|
| Projected latent replay（论文基线，11.6%） | 回放投影后的专家潜轨迹 | 同样使用动作对齐潜变量，但没有感知规划器按当前观测重规划，跨变化泛化弱 |
| Kinematic planning（论文基线，3.5%） | 运动学目标 | 规划几何目标而非 BFM 可执行潜命令，技能特有的控制响应需由下层自行补全 |
| [MotionWAM](./paper-motionwam-humanoid-loco-manipulation-wam.md) | 全身移动操作中的潜空间接口 | 同属“高层规划 + 潜空间接口”路线；FIBER 的特点是通过 LPP 将独立专家投影进冻结 BFM，再离线蒸馏统一规划器 |

上表前两行来自论文同一仿真设置（三任务、每任务 128 种变化）；与 MotionWAM 的对比是接口层面的定性区分，两者没有共享评测口径，数值不可直接比较。

## 结论

FIBER 的主要贡献是把冻结全身控制器的可执行潜变量变成可共享、可规划的技能接口，并在 G1 真机上展示了从箱子拾取到开门和抓取的统一规划。

1. 复现阅读先追踪 LPP 的动作匹配目标与 AH-BFM 潜命令定义。
2. 评估接口时同时看规划器频率和 BFM 闭环频率；5 Hz / 50 Hz 是两层职责分工。
3. 对比方法时区分 latent replay、kinematic targets 与动作对齐潜规划，避免混成“同类 planner”。
4. 引用 42.0% 时保留其仿真、三任务和每任务 128 种变化的口径。
5. 项目页已公开演示与 playground；要复现实验仍需等待可运行代码。

## 关联页面

- [Loco-Manipulation](../tasks/loco-manipulation.md) — 任务与路线总览。
- [Unitree G1](./unitree-g1.md) — 真机评测平台。
- [MotionWAM](./paper-motionwam-humanoid-loco-manipulation-wam.md) — 同在全身移动操作中使用潜空间接口的相关路线。

## 参考来源

- [FIBER 论文来源归档](../../sources/papers/fiber_arxiv_2610_24675.md)
- [FIBER 项目页归档](../../sources/sites/fiber.md)
- [FIBER 官方仓库归档](../../sources/repos/fiber.md)

## 推荐继续阅读

- [arXiv:2610.24675](https://arxiv.org/abs/2610.24675)
- [FIBER 官方项目页](https://fiberaa.github.io/)
- [FIBER 官方代码仓库](https://github.com/Yutang-Lin/FIBER)
