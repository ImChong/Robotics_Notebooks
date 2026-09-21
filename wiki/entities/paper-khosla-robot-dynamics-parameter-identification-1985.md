---
type: entity
tags:
  - paper
  - system-identification
  - robot-dynamics
  - sim2real
status: complete
updated: 2026-09-20
venue: "CDC 1985"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_01_khosla-robot-dynamics-parameter-identification-1985.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "经典机器人动力学参数辨识起点，奠定从输入–输出数据反推动力学参数的范式。"
---

# Parameter identification of robot dynamics

**Parameter identification of robot dynamics**（CDC 1985）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[01/44]**，归类 **系统辨识**。

## 一句话定义

经典机器人动力学参数辨识起点，奠定从输入–输出数据反推动力学参数的范式。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SysID | System Identification | 系统辨识 |
| CDC | Conference on Decision and Control | 控制决策会议 |
| DOF | Degrees of Freedom | 自由度 |

## 为什么重要

- 理解后续基参数、秩亏与实验设计问题的历史源头。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **系统辨识** 节点。
- 开源结论：**不适用**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | CDC 1985 |
| **文内章节** | 系统辨识 |
| **要点** | 给定关节轨迹与力矩观测，最小化模型预测与实测的动力学残差以估计参数。 |
| **开源** | **不适用** |


## 源码运行时序图

**不适用（不适用）** — 经典文献或策展资源，无可运行官方代码仓。


## 实验与评测

- **本页为索引级节点**（FreeDof 44 篇梳理 [01/44]）：正文固化文内角色与机制要点，**未转存原文实验表**。
- **回原文须核对的证据**：本页要点是「最小化模型预测与实测的动力学残差以估计参数」，对应证据是给定关节轨迹 / 力矩观测下的参数估计误差与残差收敛情况。
- **读法：** 先对齐平台、任务、指标定义与成功阈值，再读任何数字；勿从公众号摘录外推。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **文内路线** | 归类 **系统辨识**；同路线其他节点见 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) |
| **横比口径** | 1985 年的机型与传感精度决定了文内估计精度；该数字不代表现代腿足执行器（谐波 / 准直驱）上的可达精度。 |
| **开源状态** | **不适用** — 部署 / 复现前以项目页或原文 Code availability 为准 |

## 结论

**作为 SysID 文献起点阅读即可；现代腿足执行器辨识需结合 PACE 等工程约束。**

1. 文内角色：系统辨识 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：给定关节轨迹与力矩观测，最小化模型预测与实测的动力学残差以估计参数。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_01_khosla-robot-dynamics-parameter-identification-1985.md](../../sources/papers/freedof_sim2real_01_khosla-robot-dynamics-parameter-identification-1985.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
