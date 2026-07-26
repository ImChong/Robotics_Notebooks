---
type: entity
tags: [person, hardware, actuator, quadruped, locomotion, mit, boston-dynamics, physical-intelligence]
status: complete
updated: 2026-07-25
related:
  - ./mit-mini-cheetah.md
  - ./boston-dynamics.md
  - ./paper-mini-cheetah-platform.md
  - ./paper-wbic-mpc-mini-cheetah.md
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ../queries/actuator-drive-chain-selection-loop.md
  - ../../roadmap/depth-torque-motor-design.md
  - ./quadruped-robot.md
sources:
  - ../../sources/sites/robot-daycare.md
  - ../../sources/blogs/robot_daycare_mini_cheetah_2019.md
  - ../../sources/repos/bgkatz.md
summary: "Benjamin Katz（Ben）：MIT Mini Cheetah 主设计者；后入 Boston Dynamics Atlas 电动化，现 Physical Intelligence 做机器人硬件；公开博客 Robot Daycare 与 GitHub bgkatz。"
---

# Benjamin Katz（Ben Katz）

## 一句话定义

**Benjamin Katz** 是 **MIT Mini Cheetah** 的主设计者与早期交付工程师：把 hobby BLDC + 定制驱动做成可背驱模块化执行器与整机平台，并以 [Robot Daycare](https://robot-daycare.com/) / [bgkatz](https://github.com/bgkatz) 持续公开硬件工程实践；职业路径为 MIT Biomimetic Robotics Lab → Boston Dynamics Atlas → Physical Intelligence。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BLDC | Brushless DC Motor | 无刷直流电机；Mini Cheetah 执行器起点 |
| QDD | Quasi-Direct Drive | 低减速比、可背驱的力控作动思路 |
| FOC | Field-Oriented Control | 磁场定向控制；其开源驱动固件核心 |
| MPC | Model Predictive Control | Mini Cheetah 经典 locomotion 上层 |
| WBIC | Whole-Body Impulse Control | 与 MPC 配合的全身冲量控制 |

## 为什么重要

- **平台杠杆：** Mini Cheetah 成为 2019–2022 年大量 MPC/视觉/RL 论文的「公共实验床」；理解 Katz 的硬件选择（可背驱、抗冲击、单人可搬运）才能读懂后续控制论文的约束。
- **开源硬件线索：** [bgkatz](../../sources/repos/bgkatz.md) 的三相驱动、电机固件、CAN 脊柱与电源，是复现/仿制 Mini Cheetah 电气栈的一手入口。
- **产业衔接：** About 页明示 Atlas 电动化与 Physical Intelligence 硬件工作——把学术四足执行器经验接到产品级人形/通用机器人硬件。

## 核心脉络

### 1. MIT：从执行器到 Mini Cheetah

- 本科起用廉价 hobby 无刷电机做腿足；完成驱动、减速箱与 2-DoF 腿后留实验室做硕士。
- 硕士论文（[dspace:1721.1/118671](https://dspace.mit.edu/handle/1721.1/118671)）系统写 Mini Cheetah 与执行器。
- 2018 首台整机；后协助「跑通」并再建约 10 台供实验室与外借（见 [博文](../../sources/blogs/robot_daycare_mini_cheetah_2019.md)）。
- 平台论文：[Mini Cheetah Platform（ICRA 2019）](./paper-mini-cheetah-platform.md)。

### 2. Boston Dynamics Atlas（2019–2025）

- 参与电动 Atlas：电机、执行器、夹爪、结构、机构与分析等（[About](https://robot-daycare.com/about/)）。
- 与本站 [Boston Dynamics](./boston-dynamics.md) 实体叙事相接，侧重**硬件实现侧**而非公开控制论文。

### 3. Physical Intelligence（2025–）

- 现职机器人硬件（机构标签 `physical-intelligence`）。

## 工程实践

| 项 | 建议 |
|----|------|
| 读硬件 | 先博文/平台论文，再硕士论文，再 bgkatz 驱动仓 |
| 读控制 | [WBIC+MPC](./paper-wbic-mpc-mini-cheetah.md) + [Cheetah-Software](../../sources/repos/cheetah-software.md) |
| 对照开源 QDD | [open-source-qdd-actuator-projects](../comparisons/open-source-qdd-actuator-projects.md) |
| 电机设计路线 | [depth-torque-motor-design](../../roadmap/depth-torque-motor-design.md) |

## 局限与风险

- 个人博客是**工程叙事**，不是完整 BOM/供应链文档；商业敏感细节不会公开。
- Mini Cheetah 整机并非「一键开源套件」：软件栈开源度高于完整机械量产文件。
- 职业后期工作（Atlas / π）多为专有，勿从 Robot Daycare 旧文外推当前产品细节。

## 关联页面

- [MIT Mini Cheetah](./mit-mini-cheetah.md)
- [paper-mini-cheetah-platform](./paper-mini-cheetah-platform.md)
- [Boston Dynamics](./boston-dynamics.md)
- [开源 QDD 项目对比](../comparisons/open-source-qdd-actuator-projects.md)
- [执行器驱动链选型闭环知识链](../queries/actuator-drive-chain-selection-loop.md) — Ben 自研可背驱 QDD 执行器 + 驱动板固件是驱动链 **①EDA / ②FOC 固件** 层的开源范式
- [四足机器人](./quadruped-robot.md)

## 参考来源

- [Robot Daycare About](../../sources/sites/robot-daycare.md)
- [The Mini Cheetah Robot 博文](../../sources/blogs/robot_daycare_mini_cheetah_2019.md)
- [bgkatz GitHub 导航](../../sources/repos/bgkatz.md)

## 推荐继续阅读

- 个人站点：<https://robot-daycare.com/about/>
- GitHub：<https://github.com/bgkatz>
- 硕士论文：<https://dspace.mit.edu/handle/1721.1/118671>
