---
type: concept
tags: [humanoid, control, comparison, embodiment]
status: complete
updated: 2026-07-14
summary: "人形机器人因浮动基、多接触切换与全身耦合，在控制问题结构上与机械臂、轮式、四足有本质差异；飞书 Know-How 单独强调此区别。"
related:
  - ./humanoid-rubber-man-analogy.md
  - ./kinematic-vs-dynamic-feasibility.md
  - ./floating-base-dynamics.md
  - ../entities/humanoid-robot.md
  - ../overview/humanoid-motion-control-know-how-technology-map.md
sources:
  - ../../sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md
  - ../../sources/papers/humanoid_motion_control_know_how.md
---

# 人形机器人与其他机器人的区别

飞书 Know-How「人形机器人与其他机器人的区别」强调：人形不是「自由度更多的机械臂」或「双足版四足」，而是**浮动基 + 间歇接触 + 上身–下肢强耦合** 带来的独特控制问题集合。

## 一句话定义

人形难在全身动态平衡与接触切换，而不在关节数量本身。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DoF | Degrees of Freedom | 自由度数量不是唯一难点 |
| WBC | Whole-Body Control | 人形几乎必需的多任务协调层 |
| ZMP | Zero Moment Point | 双足特有平衡判据之一 |
| CoM | Center of Mass | 全身质心动力学核心 |
| RL | Reinforcement Learning | 人形 loco 常用但非唯一 |
| Sim2Real | Simulation to Real | 接触与惯量误差对人形更致命 |

## 为什么重要

- **选型与算法迁移**：四足/轮式代码不能零改动上人形。
- **硬件定义**：文档讨论「仿人 DoF」vs「波士顿动力式创新」vs「科研友好 G1」等不同人形定义。
- **知识图谱分叉**：避免把 manipulation-only 概念误当人形主干。

## 核心对比

飞书全文「一句话」：**高维度非线性、动力学突变、低静态稳定裕度的浮动基系统。**

| 对比对象 | 人形难点（全文归纳） |
|----------|---------------------|
| **无人车** | 车可近似 2D、低维低耦合、推一把不翻；人形高维串联、强非线性耦合 |
| **无人机** | 旋翼动力学无接触突变、微分平坦；人形靠接触反力、接触切换突变 |
| **四足** | 同为浮动基+突变；人形双足稳定裕度更低，且需兼顾操作与全身协调 |
| **机械臂** | 固定基、稳定裕度满；IK 即可 oftentimes 够用；人形必须处理倾倒与浮动基 |

**传统 MPC 瓶颈（作者观点）：** 不简化模型则难用现有求解器处理多接触；波士顿动力靠极致工程（离线规划+在线优化）才 demo，产出比低。

## 工程实践

- 导入人形 URDF 时显式标注 **浮动基** 与 **接触传感器** 假设。
- 从四足迁移奖励时增加 **上身姿态、双臂惯性、双足支撑多边形** 约束。
- 阅读 [橡皮人类比](./humanoid-rubber-man-analogy.md) 理解「几何像人」≠「控制像人」。

## 局限与风险

- **过度泛化**：某些人形任务（平面慢走）可用简化模型；勿一律上全 NMPC。
- **平台差异**：G1、H1、自研人形质量分布与驱动差异大，需分平台调参。

## 关联页面

- [橡皮人类比](./humanoid-rubber-man-analogy.md)
- [运动学 vs 动力学可行](./kinematic-vs-dynamic-feasibility.md)
- [Floating-Base Dynamics](./floating-base-dynamics.md)
- [Know-How 技术地图](../overview/humanoid-motion-control-know-how-technology-map.md)

## 参考来源

- [飞书 Know-How 全文](../../sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md) — §人形机器人与其他机器人的区别
- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)

## 推荐继续阅读

- [Humanoid Robot 实体](../entities/humanoid-robot.md)
