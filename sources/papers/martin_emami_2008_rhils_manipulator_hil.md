# Design and Simulation of Robot Manipulators Using a Modular Hardware-in-the-loop Platform

> 来源归档

- **标题：** Design and Simulation of Robot Manipulators Using a Modular Hardware-in-the-loop Platform
- **类型：** paper（IntechOpen 书章）
- **作者：** Adrian Martín, M. Reza Emami
- **机构：** University of Toronto Institute for Aerospace Studies（UTIAS）
- **收录：** *Robot Manipulators*（Marco Ceccarelli 编），IntechOpen，2008-09-01
- **DOI：** https://doi.org/10.5772/6214
- **章节页：** https://www.intechopen.com/chapters/5596
- **PDF：** https://api.intechopen.com/chapter/pdf-download/5596.pdf（CC BY 3.0）
- **学位论文前身：** Adrian Martin, *Development of a Modular Hardware-in-the-Loop Simulation Platform for Synthesis and Analysis of Robot Manipulators*, M.A.Sc., University of Toronto, April 2007 — https://utoronto.scholaris.ca/bitstreams/4c76372a-7936-4675-8209-5df4d6a291b9/download
- **入库日期：** 2026-09-06
- **一句话说明：** 提出面向串联机械臂的 **RHILS**（Robotic Hardware-in-the-Loop Simulation）模块化架构：用户界面、计算机仿真、硬件仿真与控制子系统四块，支持关节硬件与控制律并发设计与台架验证。
- **开源状态：** **不适用** — 学术书章与 2007 学位论文；无官方代码仓。

---

## 核心贡献（摘录）

1. **RHILS 定义：** 将机械臂部分物理组件（执行器、控制器等）接入实时仿真回路，其余动力学与环境由计算机模型承担，用于 **设计 + 测试** 一体化。
2. **四子系统架构：** (a) User Interface；(b) Computer Simulation；(c) Hardware Emulation；(d) Control System。平台组件与待测系统（Test System）分离，强调 **generic + modular**。
3. **工业臂案例：** 在标准工业机械臂（CRS CataLyst 系）及其控制器上实现 RHILS，验证正常与激进工况下的跟踪与平台可用性。
4. **与纯仿真对比：** 纯 CAD/仿真便宜但易积累建模误差；纯物理样机成本高。HIL 在二者间折中，适合 **并发工程** 与控制器早期集成测试。
5. **领域脉络：** 综述汽车 ECU、航空、制造等 HIL 应用，并聚焦 **serial-link manipulator** 的专用平台需求。

## 对 wiki 的映射

- 概念页：[Hardware-in-the-Loop（硬件在环）](../../wiki/concepts/hardware-in-the-loop.md)
- 对比 SIL：[Software-in-the-Loop](../../wiki/concepts/software-in-the-loop.md)
- 人形开发流程中的 HIL 台架：[humanoid-robot](../../wiki/entities/humanoid-robot.md)
