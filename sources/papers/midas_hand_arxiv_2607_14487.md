# MIDAS Hand: Modular low-Impedance Directly-driven Anthropomorphic Sensing Hand

> 来源归档（ingest）

- **标题：** MIDAS Hand: Modular low-Impedance Directly-driven Anthropomorphic Sensing Hand
- **类型：** paper（arXiv 预印本）
- **机构：** UCLA（加州大学洛杉矶分校）— Computer Science、Electrical and Computer Engineering、Mechanical and Aerospace Engineering；Dennis Hong 实验室
- **原始链接：**
  - <https://arxiv.org/abs/2607.14487>
  - <https://arxiv.org/pdf/2607.14487>
  - 项目页：<https://midas-hand.com>
  - 代码组织：<https://github.com/midas-hand-org>
- **入库日期：** 2026-07-20
- **一句话说明：** UCLA 开源 **直驱低阻抗仿人触觉灵巧手 MIDAS Hand**：16 总 DoF / 13 主动 DoF、283 个三轴触觉 taxel、约 700 g、BOM **<3,000 USD**、3D 打印 **<3 h** 装配；全栈发布 CAD/BOM/装配、Python 控制与触觉 API、MuJoCo、重定向与遥操作管线。

## 核心论文摘录（MVP）

### 1) 动机：灵巧操作受限于可及硬件，需同时满足形态、触觉、成本与可维护

- **链接：** <https://arxiv.org/pdf/2607.14487#section.1>
- **摘录要点：** 灵巧操作算法数据饥渴，但现有手往往在 **人形尺度、触觉、成本、可复现** 上取舍；商业手（Allegro、Wuji、Sharpa）贵且封闭，开源手（LEAP、RUKA、ORCA、CRAFT）常缺触觉或腱驱维护难。**MIDAS** 目标是在单一平台同时提供 **直驱低背驱力矩、密集三轴触觉、3D 打印模块化、全栈软件**。
- **对 wiki 的映射：**
  - [MIDAS Hand](../../wiki/entities/midas-hand.md) — 平台定位与选型对照。
  - [灵巧操作数据采集指南](../../wiki/queries/dexterous-data-collection-guide.md) — 开源触觉手 + 遥操作范例。

### 2) 硬件：直驱 Dynamixel、四指 13 主动 DoF、Paxini 283 taxel、模块化 3D 打印

- **链接：** <https://arxiv.org/pdf/2607.14487#section.3>
- **摘录要点：**
  - **尺寸** 205×120×55 mm，接近成人手；**重量** ~700 g；**BOM** <3,000 USD；装配 **<3 h**。
  - **驱动**：13 个相同 **Dynamixel XM335-T323-T** 直驱刚性连杆；指 DIP 经 **交叉四连杆** 与 PIP 欠驱动耦合（被动 DIP）。
  - **触觉**：食指/中指/无名指指尖各 **Paxini PX6AX-GEN3-DP-S2015-Elite**（52 taxel/指），拇指 **M2826-Omega**（127 taxel）；合计 **283** 三轴 taxel；最高 **83.3 Hz**。
  - **模块化**：食/中/无名指为相同可拆模块；换指 **<15 min**。
  - 论文 Table I：相对 LEAP / RUKA-v2 / ORCA / Allegro / Sharpa / Wuji，MIDAS 标注 **人形尺度 + 直驱 + DTA 触觉 + <3K USD**。
- **对 wiki 的映射：**
  - [MIDAS Hand](../../wiki/entities/midas-hand.md) — 硬件规格与对比表。

### 3) 开源生态：CAD/装配 + API + MuJoCo + 重定向/遥操作 + 数据采集

- **链接：** <https://arxiv.org/pdf/2607.14487#section.3.6>
- **摘录要点：**
  - **硬件发布**：Onshape CAD、STEP、BOM、PCB、装配指南、触觉/执行器套件。
  - **软件栈**（GitHub `midas-hand-org`）：`midas_hand_api`（Dynamixel + Paxini Python API、自动 homing）、`midas_hand_mujoco`（MJCF/URDF）、`midas_hand_retargeter`（dex-retargeting 封装）、`midas_hand_teleop`（MediaPipe 摄像头 → 仿真/真机）。
  - **遥操作输入**：视觉手部跟踪或 **MANUS 手套**；配 MANUS Haptic Pro 可将触觉映射回操作者力反馈。
  - **重定向**：指尖 IK + pinch 标定，基于 AnyTeleop 等开源重定向代码并针对 MIDAS 运动学调参。
- **对 wiki 的映射：**
  - [MIDAS Hand](../../wiki/entities/midas-hand.md) — 软件管线 mermaid 总览。
  - [Teleoperation](../../wiki/tasks/teleoperation.md) — 视觉/MANUS 双模态采集。

### 4) 实验：低背驱力矩、GRASP 32/33、载荷与可靠性

- **链接：** <https://arxiv.org/pdf/2607.14487#section.4>
- **摘录要点：**
  - **背驱力矩** 各关节约 **0.02 N·m**（同协议测 Sharpa **0.03–0.62 N·m**；Wuji 不可背驱）→ 约 **3.5–30×** 更易背驱。
  - **拇指对指工作空间** 与人手类似的桡→尺侧递减趋势（对食指/中指/无名指归一化重叠 49.5% / 40.0% / 24.5%）。
  - **强度**：指尖提 **1.2 kg**（随后电机过热）；整手抓取约 **9.5 kg**。
  - **可靠性**：2 h **5,143** 次抓握，电机稳态约 **49 °C**；100 次闭合重复性 **σ=0.016 mm**。
  - **GRASP taxonomy**：**32/33** 成功（缺小指导致 1 类失败）。
- **对 wiki 的映射：**
  - [MIDAS Hand](../../wiki/entities/midas-hand.md) — 局限（无小指、触觉未闭环控制、任务级自主未评）。

### 5) 局限与未来工作

- **链接：** <https://arxiv.org/pdf/2607.14487#section.5>
- **摘录要点：** 四指设计限制尺侧支撑；当前评的是 **硬件就绪度** 而非任务级自主操作；触觉仅作传感展示、**尚未闭环控制**；未来计划触觉系统定量表征、学习接口与触觉策略对比。
- **对 wiki 的映射：**
  - [MIDAS Hand](../../wiki/entities/midas-hand.md) — 「局限与风险」小节。

## 关联资料

- 项目页归档：[`sources/sites/midas-hand-com.md`](../sites/midas-hand-com.md)
- 代码归档：[`sources/repos/midas-hand-org.md`](../repos/midas-hand-org.md)
