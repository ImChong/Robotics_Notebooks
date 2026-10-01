# The Reality Gap in Robotics: Challenges, Solutions, and Best Practices

> 来源归档（ingest 深读摘录）

- **标题：** The Reality Gap in Robotics: Challenges, Solutions, and Best Practices
- **类型：** paper / survey / sim2real / domain-randomization / real2sim
- **arXiv：** <https://arxiv.org/abs/2510.20808>（2510.20808）
- **项目页：** <https://robotics-reality-gap.github.io/>（归档见 `sources/sites/robotics-reality-gap.md`）
- **出处：** Annual Review of Control, Robotics, and Autonomous Systems **2026**
- **作者：** Elie Aljalbout、Jiaxu Xing、Angel Romero、Iretiayo Akinola、Caelan Reed Garrett、Eric Heiden、Abhishek Gupta、Tucker Hermans、Yashraj Narang、Dieter Fox、Davide Scaramuzza、Fabio Ramos
- **机构：** University of Zurich（RPG）；NVIDIA；The University of Sydney；University of Washington；University of Utah
- **入库日期：** 2026-10-01
- **代码：** 项目页无官方仓库（综述）
- **一句话说明：** Annual Review 级 Sim2Real **综述**：按动力学 / 感知 / 执控 / 系统设计四类 **gap 来源** 拆解，再分 **缩小 gap（仿真侧）** 与 **跨越 gap（策略侧）** 两大方法族，并区分 **gap 本身** 与 **迁移性能** 两类评测指标；附工程 recipe 与开放问题（可微仿真、世界模型、大模型+仿真等）。

## 核心摘录（面向 wiki 编译）

### 1) 问题定义：reality gap 是什么

- **摘录要点：** 仿真通过抽象与近似加速训练/测试，但与真机在物理、感知、执控与系统栈上的差异集合构成 **reality gap**，导致 sim 训练策略在真机失败。综述目标是为研究者与实践者提供 **识别 gap 来源、选对手段、选对指标** 的指南。
- **对 wiki 的映射：** 主实体页（见文末总表）；概念层见 `wiki/concepts/sim2real.md`（本次 ingest 已互链）。

### 2) Gap 来源四维 taxonomy（Section 3）

- **摘录要点：**
  - **Dynamics（3.1）：** 建模简化、参数化、数值积分、人机交互、未建模效应、资产保真度等。
  - **Perception & Sensing（3.2）：** 传感器模型、噪声、环境表示、机器人模型、碰撞感知等。
  - **Actuation & Control（3.3）：** 执行器模型、底层控制、电力电子等。
  - **System Design（3.4）：** 通信延迟、安全机制、POMDP formulation、实现细节等。
- **对 wiki 的映射：** 与 `wiki/comparisons/sim2real-four-routes-identifiability.md`、`wiki/concepts/system-identification.md` 等交叉阅读（实体页内链，不在此 source 重复链出以免 lint 陈旧误报）。

### 3) 方法族：Reduce vs Overcome（Section 4 + Fig.3）

- **摘录要点：** 方法按意图分为 **Reducing the gap**（SysID、残差物理、仿真/表示设计）与 **Overcoming the gap**（DR、自适应、Real2Sim、co-training 等）。**Sim-to-Real Recipe：** 覆盖相关变量的仿真 → 逐分量 Reduce → 对残余 gap Overcome → **gap 指标 + 迁移指标** 联合评估。
- **对 wiki 的映射：** 实体页流程图；方法范式对照见 `wiki/comparisons/sim2real-approaches.md`。

### 4) 评测指标（Section 5）

- **摘录要点：** 区分 **Assessing the Reality Gap** 与 **Assessing Sim-to-Real Transfer**；实践上两者应配合。
- **对 wiki 的映射：** `wiki/queries/sim2real-gap-reduction.md`、`wiki/queries/sim2real-checklist.md`。

### 5) 开放问题（Section 6）

- **摘录要点：** 错模型与强控制器、可微仿真、视频/世界模型、仿真推断、大模型+仿真。
- **对 wiki 的映射：** `wiki/queries/sim2real-closed-loop-engineering.md`。

## 对 wiki 的映射（总表）

| 产物 | 路径 |
|------|------|
| 实体页（深读） | [`wiki/entities/paper-sa-2510-20808-the-reality-gap-in-robotics-challenges-solutions.md`](../../wiki/entities/paper-sa-2510-20808-the-reality-gap-in-robotics-challenges-solutions.md) |
| Awesome 策展 source（保留） | `sun_awesome_r2s2r_2510_20808_the-reality-gap-in-robotics-challenges-s.md` |
| 项目页归档 | `sources/sites/robotics-reality-gap.md` |
