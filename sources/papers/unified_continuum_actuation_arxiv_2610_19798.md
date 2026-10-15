# Design and Characterization of a Unified Continuum Robot Actuation Platform（arXiv:2610.19798）

> 来源归档（ingest）

- **标题：** Design and Characterization of a Unified Continuum Robot Actuation Platform
- **arXiv：** <https://arxiv.org/abs/2610.19798>（PDF：<https://arxiv.org/pdf/2610.19798>；HTML：<https://arxiv.org/html/2610.19798>）
- **提交日期：** 2026-10-14（v1，cs.RO）
- **项目页：** <https://ucsdmorimotolab.github.io/unified-actuation/>
- **代码（项目页链接）：** 硬件 <https://github.com/UCSDMorimotoLab/unified-actuation-hardware>；软件 <https://github.com/UCSDMorimotoLab/unified-actuation-software>（仓库内容与许可证本次未能访问核实）
- **相关前序工作（项目页链接）：** <https://arxiv.org/abs/2604.15638>（Contact-Aware Planning and Control of Continuum Robots in Highly Constrained Environments）
- **作者：** Yunti Xu, Aedan Mangan, Miheer Potdar, Alex Lu, Tania K. Morimoto（PI）
- **机构：** University of California San Diego（UCSD），Department of Mechanical and Aerospace Engineering（Morimoto Lab）
- **入库日期：** 2026-10-15
- **一句话说明：** 把连续体机器人的驱动硬件与软件与机械构型解耦，做成统一的模块化驱动平台；在开源手术机器人 dVRK 的 PSM 上验证精度，并演示 TDCR 相机臂与 CTR 双臂器械的远程操作。

## 开源状态（步骤 2.5，2026-10-15）

- **项目页核查：** 页面 Code 区链接上述两个 GitHub 仓库，并将平台描述为开源硬件与软件；页面未出现 "coming soon"。论文正文称项目页托管定制 CAD 与电机控制软件。
- **仓库核查：** 本次环境中 GitHub API 与仓库页面不可访问（403），仓库内容、README 与许可证未核实。
- **结论：** **已开源（按项目页声明；仓库内容未核实）**。

## 摘录 1：问题与主张（Abstract / §I）

- **痛点：** 连续体机器人（用于手术与工业检测的狭窄空间作业）的驱动系统多为单一设计定制，难以复用。
- **主张：** 将驱动硬件与软件与机械设计分离，同一套模块可驱动多类连续体机器人，支持大自由度扩展，并可与标准串联机械臂组合。

## 摘录 2：结构（论文章节）

- **机械设计（§II）：** 平移机构、旋转机构、器械驱动机构三类模块；装配与串联机械臂的机械接口；电子设计。
- **软件设计（§III）：** 电机运动控制与软件集成。
- **设备评测（§IV）：** 执行模块精度表征；运动范围表征。
- **手术远程操作（§V）：** TDCR 相机臂的扩展；CTR 镊子与常规器械的双臂遥操作；其他连续体构型的扩展讨论。

## 摘录 3：关键数值（Abstract / §IV，论文报告）

- 执行模块：30 mm 平移范围内亚毫米精度；180° 旋转范围内旋转误差小于 5°。
- 双臂实验：2.53 mm 同心管（CTR）镊子与常规腕式器械协同。
- 验证对象：开源手术机器人 dVRK，以其 Patient-Side Manipulator（PSM）作串联臂。

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-unified-continuum-actuation.md`](../../wiki/entities/paper-unified-continuum-actuation.md)。
- 与 [腱驱动软连续体 MuJoCo](../../wiki/entities/paper-quadrotor-tendon-soft-continuum-arm-mujoco.md)、[TDHD 手术双臂](../../wiki/entities/paper-tdhd-surgical-dual-arm.md) 互为相关，区分驱动平台与控制/仿真工作。
