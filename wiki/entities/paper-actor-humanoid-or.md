---
type: entity
project_id: actor-humanoid-or
tags:
  - paper
  - humanoid
  - llm-agent
  - digital-twin
  - medical
  - mobile-manipulation
status: complete
updated: 2026-11-09
arxiv: "2611.04255"
original:
  - ../../sources/papers/actor-humanoid-or_arxiv_2611_04255.md
related:
  - ./unitree-g1.md
  - ./humanoid-robot.md
  - ./anygrasp.md
  - ./mujoco.md
  - ./paper-saycan.md
  - ./paper-homebody.md
  - ./paper-loco-manip-161-135-amo.md
  - ./paper-tdhd-surgical-dual-arm.md
  - ../concepts/whole-body-control.md
  - ../concepts/llm-robotics-control-interfaces.md
  - ../concepts/humanoid-cognitive-reliability-gap.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/actor-humanoid-or_arxiv_2611_04255.md
summary: "ActOR（arXiv:2611.04255，约翰斯·霍普金斯大学）：把墙面环境相机与人形头部相机融合为共享的物体级数字孪生，由 LLM 在其上调度预编程技能并凭执行反馈重试 / 重规划；手术室仿真中闭环将成功率由 46.7% 提到 83.3%，Unitree G1 实机 20 次单瓶丢弃成功 14 次。"
---

# ActOR：手术室人形机器人的环境智能体编排（arXiv:2611.04255）

**ActOR**（*Ambient AI-Based Agentic Orchestration for Humanoid Autonomy in Operating Rooms*，[arXiv:2611.04255](https://arxiv.org/abs/2611.04255)，v1，2026-11-03，cs.RO）来自约翰斯·霍普金斯大学与 Johns Hopkins Medical Institutions（Han Zhang、Sicheng Lu 同等贡献；通讯 Mathias Unberath 组）。代码论文称「发表后公开」，目前未开源。

## 一句话定义

用手术室里已有的**墙面环境相机**补足人形机器人头部相机看不全的问题：两路观测合成一个共享的**物体级数字孪生**，**LLM 智能体**在这个孪生上调用预编程的导航 / 操作技能，技能失败或场景变化时把结果和孪生更新回灌给 LLM 重试或重规划，全程无任务专用训练（摘要、第 3 节）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OR | Operating Room | 手术室 |
| DT | Digital Twin | 数字孪生；此处指物体级场景状态 S_t（式 1） |
| LLM | Large Language Model | 论文用 GPT-6.1 Sol 做规划（3.3） |
| VLM | Vision-Language Model | 论文用 GPT-6 Luna 做物体检测（3.2、4.1） |
| SAM2 | Segment Anything Model 2 | 由 2D 框得到实例掩码（3.2） |
| AMO | Adaptive Motion Optimization | 论文引用的人形全身控制器 [22]，Navigate 用它跟踪底盘速度指令（3.4） |
| SR | Success Rate | 任务成功率（4.1） |
| DoF | Degrees of Freedom | G1 EDU 为 29 DoF（4.1） |
| SLAM | Simultaneous Localization and Mapping | 机器人用机载 LiDAR SLAM 定位（4.1） |

## 为什么重要

- **作者的问题陈述（第 1 节）：** 手术室人手紧缺，周转、取物、废弃物处理耗时且重复；术后阶段（患者已离开、环境相对可控）是自动化的现实切入点。仅用机载相机，机器人进屋后需先探索，而手术室杂乱、设备敏感，不希望多余运动；同时人形执行会出错（导航停在目标前、抓取失败），按初始场景生成的计划会失效。
- **作者的立场（第 1、2 节）：** 环境感知基础设施以往主要用于给临床团队提供信息，ActOR 把它变成机器人感知基础设施的一部分；相关的 LLM 规划工作通常只用机载感知并依赖探索。
- **我的判断：** 这是一篇系统集成 + 消融的论文，没有新的学习算法；价值在于给出「环境相机 + 机载相机 + LLM 闭环」在医疗场景的一份带数字的可行性证据，以及一个可复用的结构（共享状态 + 技能库 + 反馈回灌）。

## 核心机制

### 流程总览

```mermaid
flowchart LR
  A[墙面环境相机] --> C[VLM 检测 + SAM2 掩码 + 立体深度]
  B[人形头部相机] --> C
  C --> D[共享物体级数字孪生 S_t]
  D --> E[LLM 规划器 读取 DT JSON 与执行历史]
  E --> F[技能库 Observe Explore Navigate Pick Place]
  F --> G[AMO 全身控制 / AnyGrasp 抓取 + 运动规划]
  G --> H[技能成败与 DT 更新]
  H --> E
```

### 数字孪生（3.1、3.2、S3）

- 节点 o_i = (语义标签 l_i, 3D 位置 p_i, 3D 范围 B_i, 交互状态 h_i ∈ {free, in-hand})（式 1）。
- 检测：VLM 出框 → SAM2 出掩码 → 立体深度反投影 → 变换到共享世界系（棋盘格在环境相机与头部相机中同时观测来定义，4.1）。
- 更新：每次操作技能后及每次 Observe 调用更新，且只改动当前相机视野内的节点；重叠最多的旧节点合并，否则新建；连续深度观测发现体积空闲则删除；夹爪未完全闭合则标 in-hand（3.2）。S3 给出体素 3 cm、合并阈值（互相覆盖 40% 或一方 70% 在另一方内）、删除规则、in-hand 的夹爪开度区间 0.005–0.9，以及 in / on / near 关系的派生规则。

### 闭环编排（3.3、Algorithm 1）

规划器每步拿到指令、JSON 化的 DT 与执行历史，可调单个技能、给出多步计划或宣布完成 / 失败；多步计划顺序执行，遇失败即中断并把更新后的 DT 与历史交还规划器。重试 / 换路 / 跳过物体写在系统 prompt 里（S3「Retry is a prompt rule」），不是固定重试循环。回合在规划器宣布结束或达到 T_timeout 时终止。

### 技能实现（3.4）

| 技能 | 实现要点 |
|------|----------|
| Navigate | 环境深度投影成 2D 占用图 → 目标前臂展内的站位 → A* → 底盘速度指令 → RL 全身控制器（AMO） |
| Pick | 选离目标近的手臂；AnyGrasp 候选 + 3D 包围盒启发式候选；Pinocchio 做运动学可行性与碰撞检查；按手姿与动作幅度排序；IK 转关节指令 |
| Place | 同 Pick 流程，释放位姿在目标上方 |
| Observe / Explore | 观察并更新 DT / 沿房间地标走动用头部相机找目标 |

## 实验与结果

所有数字来自论文 4.1–4.3 与补充 S1–S3。

| 研究 | 设置 | 结果 |
|------|------|------|
| 环境感知（目标起始在视野外） | 60 次 / 方法；Robot-only 对 ActOR | 成功率 66.7%（40/60）对 80.0%（48/60）；Explore 1.05 对 0.12 次 / 试验；感知相关失败 13 → 1（Fig. 3a,b） |
| 环境感知（目标在视野内） | 同上 | 73.3%（44/60）对 80.0%（48/60）；Explore 0.92 对 0.00（Fig. 3b） |
| 闭环反馈（名义执行） | 另一组布局；Open-loop 对 ActOR | 46.7%（28/60）对 83.3%（50/60）（Fig. 3c） |
| 闭环反馈（注入扰动） | 导航误差 / 物体搬移 / 抓取偏移 5–8 cm 三选一 | 8.3%（5/60）对 51.7%（31/60）；多物收集 0/20 对 12/20 |
| 按扰动类型（Table S1） | Open-loop 对 ActOR | 导航 2/24 对 15/24；搬移 3/18 对 11/18；抓取偏移 0/18 对 5/18 |
| 实机（4.3） | Unitree G1 EDU，模拟手术室，20 次单瓶丢弃，瓶子起始在机载视野外 | 14/20（70%），成功试验均未调用 Explore；6 次失败中 5 次感知（VLM 漏检 / 误识别），1 次重试后抓取仍失败 |

- 仿真任务：单物取回、双物收集、整室复位；两个由真实房间重建的手术室；物体为四种瓶子，另有手套盒与反射锤作干扰物（S2）。
- 作者自述：逐任务置信区间重叠，需要更大样本才能确认环境感知的增益；Open-loop 同时去掉了重试与重规划，其差距是二者合并的效果（4.2）。

## 结论

**ActOR 的主要收益来自「执行反馈闭环」，环境相机的收益较小且统计上尚未确认；实机证据只有一种任务、20 次试验。**

1. **先看闭环消融**：名义执行 46.7% → 83.3%，扰动下 8.3% → 51.7%（Fig. 3c），这是论文里差距最大、最可信的一项；但 Open-loop 首次失败即终止，比较的是「有无恢复」，不是与其他闭环方案比。
2. **环境感知的真实作用是省掉探索**：Explore 调用 1.05 → 0.12 次 / 试验、感知失败 13 → 1；成功率提升只有 6.7–13.3 个百分点，作者自承置信区间重叠（4.2）。
3. **抓取偏移是短板**：按扰动类型 ActOR 抓取偏移只恢复 5/18（Table S1），远低于导航误差 15/24。
4. **实机失败主要在感知**：14/20 成功，6 次失败中 5 次是 VLM 漏检 / 误识别（4.3）；部署前优先换成针对手术室物体的闭集检测器（作者在第 5 节的建议）。
5. **部署前提**：环境相机与机器人须预先在共享世界系标定（棋盘格 + 手眼 + SLAM 变换，4.1），且实验假设执行期间无人在场；检测 / 规划依赖专有 API，临床隐私需单独解决。

## 与其他工作对比

| 维度 | ActOR | [SayCan](./paper-saycan.md) 一类 LLM 技能规划 | [HomeBody](./paper-homebody.md) |
|------|-------|------|------|
| 场景感知 | 墙面环境相机 + 头部相机融合的共享 DT | 机载感知 | 机载感知 + 自主探索 + 记忆 |
| 规划 | LLM 读 DT JSON 与执行历史，调用预编程技能 | LLM 选择预定义技能 | 人形自主探索后行动 |
| 失败处理 | 技能失败 / 场景变化回灌 LLM 重试或重规划 | 本页未核对 | 本页未核对 |

作者在第 1、2 节的定位是：相关 LLM 规划工作通常只用机载感知并依赖探索，ActOR 把原本服务临床团队的环境感知基础设施变成机器人感知的一部分。论文的实验没有直接与这些工作比较，只与自身的 Robot-only、Open-loop 两个消融基线比较（4.1）。

## 工程实践

- **硬件（4.1）：** Unitree G1 EDU（29 DoF，Dex1-1 夹爪）、头部 ZED Mini、墙面 ZED X + Jetson Orin；本地模块跑在 AMD Ryzen 9 5900HX + RTX 3080；检测与规划走 API。深度估计用 Fast-FoundationStereo。
- **标定（4.1）：** 棋盘格同时被两台相机观测以定世界系；手眼标定把头部相机配准到机器人系；机载 LiDAR SLAM 给出地图系，标定时合成出 SLAM 图系到世界系的固定变换。
- **参数（4.1、S3）：** d_reach = 0.85 m，T_timeout = 1800 s（1200 s 时向规划器告警），占用栅格 0.10 m。
- **仿真（S3）：** MuJoCo 中 G1 由 AMO 全身策略控制，环境相机 2560×1440，深度叠加 0.15 px 视差噪声；检测、分割、抓取用与实机相同的模型。
- **复现状态：** 代码、完整 planner prompt 与补充视频尚未公开（Code availability、S4）。本页只读论文，未运行任何代码。

## 局限

**论文自述（第 5 节）：**

- 检测与规划依赖专有 API 模型，临床部署有隐私问题；VLM 在杂乱场景仍不可靠，作者倾向在 OR 中用针对手术室物体训练的闭集检测器。
- 腿式人形在操作之上还要做全身平衡，带来电机过热、操作精度下降，目前只能处理较大物体；轮式底盘更稳，代价是窄空间与地面线缆通过性。
- 操作仅限 pick-and-place；开柜门、推车等需要全身协调，作者指向模仿 + 强化学习策略。
- 实验假设执行期间无人在场，所有目标物位于机器人可达工作空间内（第 4 节开头）。

**我看到的未验证之处（判断，非作者自述）：**

- 实机只有一种任务（单瓶丢弃）、20 次试验、无实机基线与扰动实验；多物体与场景变化只有定性演示、无成功率（4.3）。
- 仿真消融基线较弱：Open-loop 在首次技能失败即终止，对比的是「有无任何恢复」，没有与其他 LLM 闭环方案比较；Robot-only 对比里 ActOR 的优势在视野内条件下只有 6.7 个百分点（73.3% 对 80.0%）。
- 视野外条件下性能的上限是 80%，说明除感知外仍有其他失败来源；论文未对剩余失败做同等细粒度归因（仿真仅给了感知失败 13 → 1）。
- 环境相机需要与机器人共享世界系并预先标定，换房间 / 相机被移动时的鲁棒性未测（推测）。
- 规划器与检测器为同一厂商 API 的特定版本，结果对模型版本的敏感度未报告。

## 关联页面

- [Unitree G1](./unitree-g1.md)：实验平台
- [人形机器人总览](./humanoid-robot.md)
- [AnyGrasp](./anygrasp.md)：Pick 的抓取候选来源之一
- [MuJoCo](./mujoco.md)：仿真环境
- [SayCan](./paper-saycan.md)：LLM 规划预定义技能的经典对照
- [HomeBody](./paper-homebody.md)：论文引用 [9]，人形自主探索 + 记忆 + 行动，依赖机载探索
- [AMO 全身控制论文页](./paper-loco-manip-161-135-amo.md)：Navigate 使用的全身控制器
- [TDHD 手术双臂](./paper-tdhd-surgical-dual-arm.md)：医疗场景的另一条机器人路线
- [全身控制](../concepts/whole-body-control.md)
- [LLM 与机器人控制接口](../concepts/llm-robotics-control-interfaces.md)
- [人形认知可靠性差距](../concepts/humanoid-cognitive-reliability-gap.md)
- [Loco-manipulation](../tasks/loco-manipulation.md)

## 参考来源

- [sources/papers/actor-humanoid-or_arxiv_2611_04255.md](../../sources/papers/actor-humanoid-or_arxiv_2611_04255.md)（含核对范围、周报条目核对）
- [arXiv:2611.04255](https://arxiv.org/abs/2611.04255)

## 继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.04255)
- [SayCan](./paper-saycan.md) 与 [HomeBody](./paper-homebody.md)：对照「机载感知 + 探索」与「环境感知前置」两种路线
