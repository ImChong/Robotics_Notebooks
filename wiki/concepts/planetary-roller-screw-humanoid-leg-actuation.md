---
type: concept
tags: [humanoid, hardware, actuator, leg, linear-actuator, tesla]
status: complete
updated: 2026-05-18
related:
  - ../entities/humanoid-robot.md
  - ../tasks/locomotion.md
  - ../queries/humanoid-hardware-selection.md
  - ../entities/boston-dynamics.md
  - ./sim2real.md
  - ../concepts/humanoid-parallel-joint-kinematics.md
sources:
  - ../../sources/blogs/wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md
summary: "人形腿部用行星滚柱丝杠（PRS）把电机旋转转为直线推力、再经连杆映射为关节角，是在负载/自锁/布置与力控路径上偏向「工业实用主义」的一类执行器路线，与高动态全旋转关节路线形成典型权衡。"
---

# 人形腿部行星滚柱丝杠直线驱动（PRS 路线）

## 一句话定义

人形腿部用**行星滚柱丝杠（Planetary Roller Screw, PRS）**把电机旋转转为**直线推力**，再经**连杆机构**映射为髋/膝/踝等关节角，是在**负载密度、静态保持、纵向布置与力测量路径**上偏向「工业实用主义」的一类执行器方案；它与**高减速比旋转关节**（常见于多数量产与科研人形）在**动态带宽与传动链长度**上形成典型工程权衡。

## 为什么重要

- **硬件决定控制接口**：直线驱动 + 闭链连杆会改变等效惯量、间隙与力控路径，进而影响仿真建模、Sim2Real 与低层阻抗整定。
- **路线分歧的缩影**：同一任务（稳定行走、搬运、偶尔越障）下，「极限跑跳」与「长时间工况能耗/成本」优化目标不同，会导向不同的腿部拓扑与执行器形态。
- **阅读公开报道时的锚点**：媒体常把 Tesla Optimus 与 PRS 绑定叙述；理解 PRS 能读明白「在优化什么」，避免把演示级结论直接当成普适最优解。

## 核心结构 / 机制

### 行星滚柱丝杠在做什么

- 功能上仍是**旋转–直线**变换器：电机输出转速/力矩，经丝杠副变为丝杠（或螺母）的轴向力与位移。
- 与滚珠丝杠相比，PRS 用**多枚螺纹滚柱**分担载荷，啮合面积大，因而在**同外径下的承载能力**与**疲劳寿命**上往往更有优势（具体倍数依赖几何与材料，需看厂商曲线）。

### 反转式布置（叙事中常见）

- **传统**：丝杠转、螺母直线动。
- **反转式**：**螺母由转子侧驱动旋转**，**丝杠输出直线推力**——便于把电机–丝杠做成紧凑执行器包，并把力传感器布置在更接近负载输出端的位置。

### 与关节角的接口：为什么常看到「直线 + 连杆」

- 直线执行器适合沿**大腿/小腿腔体**纵向布置；但膝关节需要大范围角位移，纯直线推杆难以单轴等效为人体膝关节运动学。
- 工程上常用**四连杆（four-bar）**等闭链机构，把有限行程的直线运动映射为关节角变化，并在特定角度区间匹配力矩需求曲线。

## 流程总览（从电机到足端）

```mermaid
flowchart LR
  M[电机 / 逆变器] --> PRS[行星滚柱丝杠副<br/>旋转 → 直线]
  PRS --> F[力传感 / 导向件]
  F --> L[连杆机构<br/>直线 → 关节角]
  L --> J[膝 / 踝等关节输出]
  J --> C[足端接触与负载]
```

## 与旋转关节路线的对比（概念层）

| 维度 | PRS + 连杆（文中叙事侧重） | 高动态旋转关节（谐波 / 行星 + 电机） |
|------|---------------------------|-------------------------------------|
| 静态保持 | 可追求机械自锁，减少站立维持电流 | 常需持续力矩或抱闸 |
| 负载密度 | 强调轴向承载与同径对比优势 | 依赖减速器扭矩密度与热约束 |
| 动态带宽 | 传动链长，极限跑跳通常不占优 | 更适合高速变向与爆发动作 |
| 建模与标定 | 闭链几何 + 丝杠非线性需仔细进仿真 | 串联链相对直观，但摩擦与柔性仍难 |

## 常见误区或局限

- **把媒体解读当官方规格**：公开活动与第三方文章不等于冻结的 BOM；量产迭代可能更换丝杠副、连杆参数或混合方案。
- **把「示意能耗表」当普适结论**：站立/慢走占比一旦改变，系统级能耗排序可能完全改写。
- **忽略闭链控制复杂度**：四连杆引入耦合与奇异性管理，WBC 与仿真资产必须与机构一致（参见 [人形机器人并联关节解算](./humanoid-parallel-joint-kinematics.md) 的分层讨论）。

## 与其他页面的关系

- [人形机器人](../entities/humanoid-robot.md)：平台级硬件与任务语境。
- [Locomotion](../tasks/locomotion.md)：行走闭环中「关节接口」与 Sim2Real 约束。
- [人形机器人并联关节解算](./humanoid-parallel-joint-kinematics.md)：闭链踝/膝在仿真与控制上的接口注意点。
- [Sim2Real](./sim2real.md)：执行器非线性与摩擦对迁移的影响。
- [Query：人形机器人硬件怎么选](../queries/humanoid-hardware-selection.md)：在多种路线间做决策时的检查清单入口。

## 推荐继续阅读

- [Rollvis 行星滚柱丝杠介绍（厂商技术页）](https://www.rollvis.com/) — 典型 PRS 结构术语与选型维度（英文站点，可作名词对齐）。
- Boston Dynamics 对 Atlas 全电化与液压路线的公开技术叙述（与「极限动态」叙事对照）：建议从 [Boston Dynamics 官网新闻/技术稿](https://www.bostondynamics.com/) 检索 *Atlas* 获取一手时间线。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 参考来源

- [特斯拉人形机器人腿部关节为什么选择行星滚柱丝杠？（微信公众号原文）](https://mp.weixin.qq.com/s/webqJRQJREZdABw8bdl68w)
- [wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md（仓库内归档）](../../sources/blogs/wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md)
