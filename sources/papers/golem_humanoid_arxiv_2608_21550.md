# GOLEM（模块化人形电池拆解）

> 来源归档（ingest）

- **标题：** GOLEM: Modular Humanoid Autonomy Towards Electric Vehicle Battery Disassembly
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.21550>
- **机构：** 科罗拉多大学博尔德分校（University of Colorado Boulder）；圣母大学（University of Notre Dame）
- **作者：** Max Conway、William Xie、Allen Devaraj、Yutong Zhang、Niraj Pudasaini、Mateo Feit、Adam Abid、Zachary Allen、Chen Liu、Xuan Tan、Jensen Lavering、Jason Chen、Lyle Antieau、Anthony Von Pischke、Alessandro Roncone、Zachary Sunberg、Nikolaus Correll
- **项目页：** <https://golem-humanoid.github.io>
- **入库日期：** 2026-08-30
- **一句话说明：** 面向 Unitree H1-2 的模块化开源架构，把行走、操作、动态稳定、导航和空间记忆拆成可替换接口，用于退役电动车电池拆解。

## 核心摘录（MVP）

### 1) 能力阶梯而不是整系统黑箱

- **摘录要点：** 行走 / 操作 / 稳定 / 导航 / 空间记忆各自抽象接口，方法可互换比较。Docker 化 ROS 2 连接 MuJoCo、IsaacLab 数字孪生与真机，切换仿真/硬件等于选 DDS 域。
- **对 wiki 的映射：**
  - [GOLEM](../../wiki/entities/paper-golem-humanoid.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 2) 能力阶梯数字

- **摘录要点：** LiDAR–惯性导航 6 m 目标误差 **13.0 cm**；学习式站立控制器能恢复采样式下肢 MPC 无法应对的扰动（项目页写对抗 RL 恢复 93%）；真实 Ioniq 5 电池包紧固件抓取：系留 **97%** → 自由站立 **87%** → 加入导航位姿扰动 **37%**。
- **对 wiki 的映射：**
  - [GOLEM](../../wiki/entities/paper-golem-humanoid.md)

### 3) 开源状态（截至 2026-08-30）

- **摘录要点：** 论文写 Source code is available at the project page；项目页列出 <https://github.com/golem-humanoid>。本环境对该 org 的 GitHub API 返回 404，**按宣称将开源 / 待核实** 处理，勿写成已可 clone 复现。

## 当前提炼状态

- [x] 项目页与 arXiv 摘要对齐
- [x] 开源边界已写入
- [x] wiki 映射：`wiki/entities/paper-golem-humanoid.md` 新建
