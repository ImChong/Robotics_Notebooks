---
type: entity
tags: [hardware, actuator, qdd, open-source, vesc, belt-drive]
status: complete
updated: 2026-07-25
related:
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ./stanford-doggo-and-pupper.md
  - ./odri-solo-and-bolt.md
  - ./vesc.md
  - ./moteus.md
  - ./simplefoc.md
  - ./ironless-qdd-actuator.md
  - ../../roadmap/depth-torque-motor-design.md
  - ../queries/actuator-drive-chain-selection-loop.md
sources:
  - ../../sources/repos/opentorque_actuator.md
  - ../../sources/repos/vesc_bldc.md
  - ../../sources/personal/open_source_qdd_actuator_learning_curator.md
summary: "OpenTorque Actuator：经典开源准直驱关节（大尺寸外转子航模电机 + 低减速同步带 + VESC + 编码器 + 3D 打印结构），适合快速做 QDD 样机；STEP/STL/BOM 公开，成熟度与抗冲击不及学术/工业关节。"
---

# OpenTorque Actuator（开源准直驱关节）

## 一句话定义

**OpenTorque Actuator**（[G-Levine/OpenTorque-Actuator](https://github.com/G-Levine/OpenTorque-Actuator)，[Hackaday](https://hackaday.io/project/159404-opentorque-actuator)）是经典 DIY **QDD** 关节：大尺寸外转子航模电机 + 低减速同步带 + **[VESC](./vesc.md)** + 编码器 + 3D 打印结构，定位为腿足用强力、可柔顺执行器。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| VESC | Vedder Electronic Speed Controller | 开源大电流 BLDC/FOC 驱动生态 |
| FOC | Field-Oriented Control | 无刷电机的磁场定向控制 |
| BLDC | Brushless DC Motor | 无刷直流电机 |
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |
| STL | Stereolithography file | 常见三角网格 3D 打印交换格式 |

## 为什么重要

- 设计叙事接近早期 Cassie / MIT Cheetah 类：**高扭矩密度电机 + 低减速 + 电流力矩控制**。
- 学习阶梯里适合作为 **第一个能转起来的低减速比关节**（在 SimpleFOC/moteus 之后、ODRI 之前）。
- 仓库直接给出 **STEP / STL / BOM / 爆炸图**，改机与复刻成本低。

## 开源资产（仓库布局）

| 路径/文件 | 内容 |
|-----------|------|
| `STEP/` | 装配/零件 STEP |
| `STL/` | 可打印件 |
| `Bill of Materials.csv` | BOM |
| `Print Instructions.txt` | 打印说明 |
| `images/` | 含 exploded view |
| Hackaday 项目页 | 装配叙事与迭代讨论 |

**开源状态：** **已开源**（机械为主；驱动依赖 [VESC](./vesc.md) 生态）。GitHub license 字段曾为 `NOASSERTION`，仓库含 `LICENSE` 文件，使用前自行核对。

## 核心结构/机制

```mermaid
flowchart LR
  mot["大尺寸外转子\n航模 BLDC"]
  belt["同步带\n低减速比"]
  out["关节输出"]
  vesc["VESC FOC"]
  enc["编码器"]
  mot --> belt --> out
  vesc --> mot
  enc --> vesc
```

| 模块 | 角色 |
|------|------|
| 电机 | 大外转子航模电机提供高 \(K_t\) / 扭矩密度 |
| 传动 | 同步带低减速：放大力矩、保留反驱与冲击容忍 |
| 驱动 | VESC 电流/力矩模式 |
| 结构 | 3D 打印外壳与支架，快速迭代 |

## 工程实践

| 步骤 | 做什么 |
|------|--------|
| 1 | 按 BOM 采购电机、带、轴承、VESC、编码器 |
| 2 | 按 Print Instructions 打件并装配；对照爆炸图对位 |
| 3 | VESC Tool 做电机辨识、电流限与编码器对齐 |
| 4 | 台架测峰值/连续力矩与温升；再上腿/摆臂 |
| 5 | 进阶对照 [ODRI](./odri-solo-and-bolt.md) 双编码器与专用驱动 PCB |

## 局限与风险

- 体积、重量、轴承布置与抗冲击通常达不到成熟人形要求。
- 同步带张紧、磨损与热管理需自行工程化。
- VESC 非关节专用总线栈——多轴 1 kHz 确定性要另做系统验证。

## 关联页面

- [开源 QDD 执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md)
- [VESC](./vesc.md)
- [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md)
- [Stanford Doggo](./stanford-doggo-and-pupper.md)（同类同步带 QDD + ODrive）
- [ODRI](./odri-solo-and-bolt.md)
- [moteus](./moteus.md)

## 参考来源

- [sources/repos/opentorque_actuator.md](../../sources/repos/opentorque_actuator.md)
- [sources/repos/vesc_bldc.md](../../sources/repos/vesc_bldc.md)
- [开源 QDD 执行器学习策展](../../sources/personal/open_source_qdd_actuator_learning_curator.md)

## 推荐继续阅读

- <https://github.com/G-Levine/OpenTorque-Actuator>
- <https://hackaday.io/project/159404-opentorque-actuator>
