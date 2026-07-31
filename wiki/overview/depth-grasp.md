---
type: overview
tags: [depth, depth-grasp, manipulation, dexterous, grasp]
status: complete
updated: 2026-07-31
summary: "抓取与操作感知纵深汇总：从接触建模、灵巧手运动学到 GraspNet/AnyGrasp 等感知抓取栈，覆盖 pick-place、双手协作与 loco-manip 中的操作子问题。"
---

# 抓取与操作（纵深汇总）

> **图谱纵深视图**：本页是知识图谱「🤏 抓取 (Grasp)」纵深的统一入口；在 [图谱纵深视图](../../docs/graph.html?depth=grasp) 筛选时，本节点为汇总锚点。

## 一句话定义

**抓取纵深** 关注机器人如何通过感知与规划，在接触丰富的环境中稳定地 **抓取、持握并操纵物体**，是人形 loco-manip 与桌面操作的核心子栈。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Grasp | Grasping | 建立并维持稳定抓取 |
| Manip | Manipulation | 抓取后的位姿/力控操作 |
| Dex | Dexterous | 多指灵巧手的高维控制 |
| CuRobo | CUDA Robot Motion | GPU 加速的运动规划栈（抓取场景常用） |
| HOI | Human-Object Interaction | 人-物交互数据与技能 |

## 为什么重要

- **人形价值在「能干活」**：Locomotion 解决去哪，抓取解决拿什么、怎么放。
- **接触是主要不确定性来源**：滑移、形变、遮挡让纯位置控制不够。
- **感知-规划-控制需闭环**：从点云/深度到抓取姿态，再到力位混合执行。

## 本纵深覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 任务 | 操作任务定义与挑战 | [Manipulation](../tasks/manipulation.md) |
| 概念 | 接触丰富操作 | [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md) |
| 运动学 | 灵巧手/多指 | [Dexterous Kinematics](../concepts/dexterous-kinematics.md) |
| 感知抓取 | 6-DoF 抓取检测与选型 | [AnyGrasp vs GraspNet](../comparisons/anygrasp-vs-graspnet.md) |
| 组合 | loco-manip 中的操作 | [Loco-Manipulation](../tasks/loco-manipulation.md) |
| 硬件 | 低成本开源三指末端 | [EN02-OP](../entities/en02-op.md)（Westwood，7 DoF + Dynamixel） |
| 硬件 | 开源欠驱动腱驱手族 | [Yale OpenHand](../entities/yale-openhand.md)（T/T42/O/F3；CAD CC BY-NC；F3 腕相机估力） |
| 数据 | 同物体人–机配对灵巧抓取 | [HRDexDB](../entities/hrdexdb-dataset.md)（100+ 物体 · 多灵巧手 · 3D + 触觉） |
| 方法 | 移动高速灵巧抓取 + 全身 RL | [FastGrasp](../entities/paper-fastgrasp-mobile-dexterous-grasping.md)（CVAE 引导 · 二值触觉 · arXiv:2604.12879） |
| 末段精修 | 纯触觉目标条件 regrasp | [TacRefineNet](../entities/paper-tacrefinenet-tactile-grasp-refinement.md)（板/盘/杆 · Siamese · arXiv:2509.25746） |

## 与其他纵深的关系

- **[触觉](./depth-tactile.md)**：抓取稳定常依赖力/触觉反馈。
- **[WBC](./depth-wbc.md)**：全身协调下手臂与躯干的分工。
- **[VLA](./depth-vla.md)**：语言条件下的抓取与放置。

## 关联页面

- [Hybrid Force-Position Control](../concepts/hybrid-force-position-control.md)
- [Impedance Control](../concepts/impedance-control.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)
- [TacRefineNet（论文实体）](../entities/paper-tacrefinenet-tactile-grasp-refinement.md) — 抓取末段纯触觉精修

## 参考来源

- 本库归纳自 [Manipulation 任务页](../tasks/manipulation.md)、[接触丰富操作](../concepts/contact-rich-manipulation.md)、[AnyGrasp vs GraspNet](../comparisons/anygrasp-vs-graspnet.md)
- **ingest 档案：** [sources/papers/tacrefinenet_arxiv_2509_25746.md](../../sources/papers/tacrefinenet_arxiv_2509_25746.md) — TacRefineNet（arXiv:2509.25746）
- 图谱纵深定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`grasp` 命中规则）
