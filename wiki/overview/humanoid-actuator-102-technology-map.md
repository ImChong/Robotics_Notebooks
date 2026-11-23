---
type: overview
tags: [humanoid, actuator, backdrivability, reflected-inertia, qdd, sea, survey]
status: complete
updated: 2026-11-21
related:
  - ./humanoid-actuator-102-load-and-mass-spiral.md
  - ./humanoid-actuator-102-split-architecture.md
  - ./humanoid-actuator-102-gear-reflected-inertia.md
  - ./humanoid-actuator-102-thermal-and-control.md
  - ./humanoid-actuator-102-compliance-sensing.md
  - ./humanoid-actuator-102-industrial-actuator-trap.md
  - ./humanoid-actuator-102-decision-species.md
  - ./humanoid-actuator-102-future-artificial-muscle.md
  - ./humanoid-hardware-101-technology-map.md
  - ./humanoid-hardware-101-integrated-actuators.md
  - ../queries/humanoid-hardware-selection.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
  - ../../sources/papers/humanoid_actuator_102_reference_catalog.md
original:
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
summary: "human five《Humanoid 执行器 入门 102》：从行走冲击与质量惩罚螺旋出发，经旋转-直线分离架构、N² 反射惯量、热控与力矩环，到工业陷阱、三大物种决策矩阵与关节指标；姊妹篇 Hardware 101 补部件 BOM。"
---

# Humanoid 执行器 102：八章技术地图

> **本页定位**：为 [human five · Humanoid 执行器 入门 102](https://mp.weixin.qq.com/s/zinp6ulTorzfqmCR_HaI5A) 提供 **按论证链条组织的阅读坐标**；部件级清单与供应链见姊妹篇 [Humanoid Hardware 101](./humanoid-hardware-101-technology-map.md)。

## 一句话定义

人形腿部执行器不是「更大扭矩的工业伺服」，而是在 **亚毫秒落地冲击** 下可机械退让、在 **质量惩罚螺旋** 下保持 `>15 Nm/kg` 峰值比力矩，并在 $N^2$ 反射惯量与连续热极限之间按整机任务谱系选对谐波、行星滚柱丝杠、QDD 或 SEA 构型的关节动力学系统。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| SEA | Series Elastic Actuator | 串联弹性执行器，提供柔顺与力控 |
| CoT | Cost of Transport | 单位重量·距离能耗的无量纲移动效率指标 |

## 为什么重要

执行器决定了人形机器人的行走寿命、跌倒耐受度、力控带宽以及约半数整机 BOM 成本。若从传统工业机械臂的“定位精度优先”视角选型，直接采购高减速刚性工业伺服会在步行每小时 5000 次冲击与平方律惯量放大中迅速损毁。本技术地图将原长文的 12 个章节压缩为一条从第一性物理约束（冲击与质量螺旋）到具体关节硬指标与三大整机物种选择的完整知识导航骨架。

## 核心原理与技术分工

### 与 Hardware 101 的分工

| 维度 | [Hardware 101](./humanoid-hardware-101-technology-map.md) | 本页（Actuator 102） |
|------|-----------------------------------------------------------|----------------------|
| 视角 | 七类子系统、BOM、供应链 | 执行器在腿上**为何坏**、如何选型 |
| 深度 | 电机/减速/丝杠/轴承分件 | 冲击、CoT、反射惯量、控制带宽 |
| 结论 | 2 万美元需减关节与简化手 | 三大物种 + 膝关节样本指标 |

## 流程总览：八章论证链

```mermaid
flowchart LR
  subgraph G1["01 负载与质量"]
    A1["5000步冲击<br/>CoT 螺旋"]
  end
  subgraph G2["02 分离架构"]
    A2["谐波旋转<br/>滚柱直线"]
  end
  subgraph G3["03 减速惯量"]
    A3["N2 反射惯量<br/>透明性光谱"]
  end
  subgraph G4["04 热与控制"]
    A4["热极限<br/>FOC 力矩环"]
  end
  subgraph G5["05 柔顺感知"]
    A5["SEA 退让<br/>双编码器"]
  end
  subgraph G6["06 工业陷阱"]
    A6["平方律<br/>货架失效"]
  end
  subgraph G7["07 决策指标"]
    A7["三大物种<br/>15Nm每kg"]
  end
  subgraph G8["08 未来"]
    A8["人工肌肉<br/>电磁瓶颈"]
  end
  A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7 --> A8
```

## 八类分类（各组一页）

| 组 | 分类页 | 公众号章节 |
|----|----------|------------|
| 01 负载与质量螺旋 | [负载与质量螺旋](./humanoid-actuator-102-load-and-mass-spiral.md) | I–II |
| 02 分离架构 | [旋转-直线分离架构](./humanoid-actuator-102-split-architecture.md) | III |
| 03 减速与反射惯量 | [减速与反射惯量](./humanoid-actuator-102-gear-reflected-inertia.md) | IV |
| 04 热学与控制 | [热学与力矩控制](./humanoid-actuator-102-thermal-and-control.md) | V–VI |
| 05 柔顺与感知 | [柔顺与感知反馈](./humanoid-actuator-102-compliance-sensing.md) | VII–VIII |
| 06 工业陷阱 | [工业执行器陷阱](./humanoid-actuator-102-industrial-actuator-trap.md) | IX |
| 07 决策与指标 | [决策矩阵与物种](./humanoid-actuator-102-decision-species.md) | X–XI |
| 08 未来路线 | [人工肌肉与未来](./humanoid-actuator-102-future-artificial-muscle.md) | XII + 参考文献 |

## 工程实践

- **按论证链顺次排查设计冲突**：先确定整机重量与着地冲击能量谱（01–02），再分配减速比并控制远端反射惯量 `<0.1 kg·m²`（03），随后校核连续蹲起温升与峰持比 `>=3:1`、FOC 力矩带宽 `50–100 Hz`（04–05），最后对照三大目标场景物种冻结关节硬指标（06–07）。
- **对接单电机与供应链底层设计**：确认系统级关节规格后，直接转入 [电机设计流程](./motor-design-workflow.md) 与 [力矩电机设计纵深路线](../../roadmap/depth-torque-motor-design.md) 完成无框电机磁路、TN/TI 包络校核与电磁仿真选型。

## 常见误区或局限

- **误区：存在单一“最强通用执行器”可以装遍全身** — 正如“没有免费午餐”决策矩阵所示，工厂搬运、敏捷跑跳与安静家用处于物理权衡三角的不同顶点，任何硬件配置都是在力矩密度、反射惯量、抗冲击与可控性之间做系统交换。
- **时效性局限**：原文（2026 年初整理归档，含大量产业实测经验量级）中对各商业人形具体规格的归纳反映当下工程认知，具体新型号仍需结合最新实机拆解或论文一手参数交叉核验。

## 关联页面

- [力矩电机设计纵深路线（Stage 0–6 学习顺序展开版）](../../roadmap/depth-torque-motor-design.md)
- [人形整机硬件设计纵深路线（关节模组之上的整机层）](../../roadmap/depth-humanoid-hardware-design.md)
- [电机设计流程（规格 → 仿真 → 样机 → 控制）](./motor-design-workflow.md)
- [电机转矩-转速曲线（TN 曲线）](../concepts/motor-torque-speed-curve.md) — 关节模组 datasheet 读图
- [电机转矩-电流曲线（TI 曲线）](../concepts/motor-torque-current-curve.md)
- [电机电磁仿真软件选型](../comparisons/motor-em-simulation-software.md)
- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [Hardware 101 · 集成执行器](./humanoid-hardware-101-integrated-actuators.md)
- [Hardware 101 · 传动链](./humanoid-hardware-101-actuation-sensing-chain.md)
- [人形硬件选型 Query](../queries/humanoid-hardware-selection.md)
- [参考文献索引（source）](../../sources/papers/humanoid_actuator_102_reference_catalog.md)

## 推荐继续阅读

- [Actuator 102 · 01：负载与质量螺旋](./humanoid-actuator-102-load-and-mass-spiral.md) — 论证链起点
- [Actuator 102 · 07：决策矩阵与三大物种](./humanoid-actuator-102-decision-species.md) — 落地规格表

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md) — <https://mp.weixin.qq.com/s/zinp6ulTorzfqmCR_HaI5A>
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)
- [humanoid_actuator_102_reference_catalog.md](../../sources/papers/humanoid_actuator_102_reference_catalog.md)
