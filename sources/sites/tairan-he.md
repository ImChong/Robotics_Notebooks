# Tairan He（何泰然）个人学术主页 — tairanhe.com

- **类型**：学者个人站点 / 履历与论文列表（原始资料归档）
- **收录日期**：2026-05-14（初收）；**2026-06-25 复核更新**
- **主链接**：<https://tairanhe.com/>

## 一句话

**OpenAI** Member of Technical Staff（*building physical AGI*）；CMU Robotics Institute 博士（2026-04 答辩，导师 Guanya Shi、Changliu Liu），曾在 NVIDIA GEAR Lab 实习约两年；研究主线为 **人形机器人的规模化学习、视觉 Sim2Real 与全身 loco-manipulation**；站点列出 VIRAL、DoorMan、OmniH2O、HOVER、ASAP、SONIC 等论文与项目入口。

## 为什么值得保留

- **高信号索引页**：集中给出项目页、arXiv BibTeX 与会议标注，便于把「CMU LECAR / NVIDIA GEAR / OpenAI 人形学习管线」与仓库已有 [Sim2Real](../../wiki/concepts/sim2real.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[遥操作](../../wiki/tasks/teleoperation.md) 等页面串起来。
- **职业节点可追溯**：主页 **首段自述** 与新闻栏写明 **2026-05 加入 OpenAI**、**2026-04 博士答辩**（答辩委员会含 Guanya Shi、Changliu Liu、Kris Kitani、Marco Hutter、Pieter Abbeel）。

## 履历要点（来自主页自述，2026-06-25 抓取）

- **当前**：OpenAI **Member of Technical Staff**，表述为 *building physical AGI*；新闻栏 **05/2026** 写明 *Joined OpenAI to work on general-purpose robots!*
- **博士**：Carnegie Mellon University Robotics Institute（**2026-04 答辩**）；导师 Guanya Shi、Changliu Liu；曾在 NVIDIA GEAR Lab（Jim Fan、Yuke Zhu）实习约两年；获 CMU RI Presidential Fellowship、NVIDIA Graduate Fellowship。
- **本科**：上海交通大学计算机（导师 Weinan Zhang）；曾在 Microsoft Research Asia 实习。
- **自述目标**：*Goal: Robots that improve everyone's life.*
- **自述研究交集**：robotics、large-scale machine learning、general-purpose loco-manipulation。
- **自述核心问题**：如何构建可扩展的 **robot learning flywheel**，统一 perception、whole-body control、dexterous manipulation，使通用人形在杂乱真实环境中可靠工作。

## 新闻与时间线（主页 News 区块）

| 时间（站点标注） | 事件 |
|------------------|------|
| 05/2026 | 加入 OpenAI，从事通用人形机器人方向 |
| 04/2026 | CMU 机器人学博士答辩（委员会：Guanya Shi、Changliu Liu、Kris Kitani、Marco Hutter、Pieter Abbeel） |
| 09/2025 | GRASP SFI Seminar 邀请报告：Scalable Sim-to-Real Learning for General-Purpose Humanoid Skills |
| 08/2025 | 通过博士论文开题答辩（Thesis Proposal） |
| 12/2024 | NVIDIA Graduate Fellowship |
| 11/2024 | CMU RI Presidential Fellowship |
| 07/2024 | ABS 入选 RSS 2024 Outstanding Student Paper Award Finalist |

## 论文与项目入口（按主页「Publications」区块归纳）

下列 **标题、会议与官方项目 URL 以主页为准**；摘要级文字为 ingest 时的压缩归纳，精读请以论文 PDF / 项目页为准。

| 主题侧重 | 代表工作（主页列出） | 会议 / 入口（示例） |
|----------|----------------------|---------------------|
| 视觉 Sim2Real、门把手 loco-manipulation | DoorMan（Opening the Sim-to-Real Door…） | **CVPR 2026**；<https://doorman-humanoid.github.io/>；arXiv:2512.01061 |
| 视觉 Sim2Real、规模化人形 loco-manipulation | VIRAL | **CVPR 2026**；<https://viral-humanoid.github.io/>；arXiv:2511.15200 |
| 规模化运动跟踪 / 通用人形低层控制 | SONIC（合作者一作，共同署名） | arXiv:2511.07820；<https://nvlabs.github.io/GEAR-SONIC/> |
| 人视频 → 交互全身技能 | HDMI | arXiv:2509.16757；<https://hdmi-humanoid.github.io/#/> |
| 力自适应 loco-manipulation | FALCON | **L4DC 2026**；arXiv:2505.06776；<https://lecar-lab.github.io/falcon-humanoid/> |
| 视觉驱动的全身灵巧（仿真） | PDC | arXiv:2505.12278；<https://www.zhengyiluo.com/PDC-Site/> |
| 人–人形跨本体模仿 | Humanoid Policy ~ Human Policy | CoRL 2025；arXiv:2503.13441；<https://human-as-robot.github.io/> |
| 主动探索式系统辨识 Sim2Real | SPI-Active | CoRL 2025；arXiv:2505.14266；<https://lecar-lab.github.io/spi-active_/> |
| 慢–快双智能体、端部稳定行走 | SoFTA（Hold My Beer） | CoRL 2025 / RSS 2025 Workshop；<https://lecar-lab.github.io/SoFTA/> |
| 仿真–真机动力学对齐（残差动作等） | ASAP | **RSS 2025**；<https://agile.human2humanoid.com/> |
| 多模式全身神经控制蒸馏 | HOVER | ICRA 2025；arXiv:2410.21229；<https://hover-versatile-humanoid.github.io/> |
| 敏捷与安全折中扩展 | BAS（相对 ABS 系列） | L4DC 2025；arXiv:2501.04276；<https://adaptive-safe-locomotion.github.io/> |
| 人–人形全身遥操作与学习 | OmniH2O | CoRL 2024；arXiv:2406.08858；<https://omni.human2humanoid.com/> |
| 顺序接触全身 RL | WoCoCo | CoRL 2024 Oral；arXiv:2406.x；<https://lecar-lab.github.io/wococo/> |

（主页上尚有更多条目与合著工作；本表只覆盖 ingest 时用于映射知识库主线的子集。）

## 对 wiki 的映射

- 升格页面：[wiki/entities/tairan-he.md](../../wiki/entities/tairan-he.md)
- 关联论文实体：[VIRAL](../../wiki/entities/paper-viral-humanoid-visual-sim2real.md)、[DoorMan](../../wiki/entities/paper-doorman-opening-sim2real-door.md)、[GR00T-VisualSim2Real](../../wiki/entities/gr00t-visual-sim2real.md)

## 参考链接

- 个人主页：<https://tairanhe.com/>
