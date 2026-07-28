---
type: entity
tags:
  - software
  - simulation
  - multibody-dynamics
  - mbd
  - industrial-cae
  - university-of-michigan
  - cadence
  - msc-software
status: complete
updated: 2026-07-28
related:
  - ./mujoco.md
  - ./drake.md
  - ./motrix.md
  - ./pybullet.md
  - ../overview/sim-platforms-decade-technology-map.md
  - ../queries/simulator-selection-guide.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/adams_orlandea_primary_refs.md
  - ../../sources/sites/umich-deepblue-orlandea-adams-thesis.md
  - ../../sources/sites/cadence-msc-adams.md
  - ../../sources/blogs/janevic_orlandea_adams_memorial.md
summary: "ADAMS（Automatic Dynamic Analysis of Mechanical Systems）是 Orlandea 1973 密歇根博士工作起源、1977 ASME 稀疏矩阵+刚性积分方法论文定名的工业多体动力学程序谱系；当代商业产品为 Cadence MSC Adams，未开源，定位整机虚拟样机而非 RL 并行训练。"
---

# ADAMS（Automatic Dynamic Analysis of Mechanical Systems）

**ADAMS** 是面向三维机械系统的 **多体动力学（Multibody Dynamics, MBD）** 自动建模与数值仿真程序谱系：名称与核心数值配方来自 Nicolae Orlandea 在密歇根大学的博士工作与 1977 ASME 方法论文；经 Mechanical Dynamics, Inc.（MDI）商业化后，沿 MSC Software / Hexagon 传承至今日的 **Cadence MSC Adams**。

## 一句话定义

用 **稀疏 tableau + 刚性（BDF）积分** 自动求解带约束的三维机械系统动力学，服务工业整机虚拟样机验证——不是开源 RL 物理引擎。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ADAMS | Automatic Dynamic Analysis of Mechanical Systems | Orlandea 程序名；今亦写作 Adams（产品） |
| MBD | Multibody Dynamics | 多刚体/柔体约束系统动力学 |
| STF | Sparse Tableaux Formulation | 稀疏 tableau 列式（作者史述核心） |
| BDF | Backward Differentiation Formula | Gear 刚性积分族；早期 ADAMS 数值底座 |
| DAE | Differential–Algebraic Equation | 运动方程 + 代数约束的混合系统 |
| NVH | Noise, Vibration, and Harshness | 当代 Adams 产品线中的整车声学/振动分析能力 |
| K&C | Kinematics & Compliance | 悬架运动学与柔度试验/仿真 |
| HIL | Hardware-in-the-Loop | Adams Real Time 等实时联仿场景 |

## 为什么重要

- **工业 MBD 谱系原点：** 1977 年 ASME 双篇明确把稀疏矩阵与刚性积分算法 **实现进名为 ADAMS 的程序**，并以 Malibu 前悬架、Boeing 747 起落架对照实验——这是「可交付的三维机构动力学软件」而非纯理论笔记。
- **与机器人学习仿真的分工：** [MuJoCo](./mujoco.md) / [Drake](./drake.md) / Isaac 等服务 **策略学习、轨迹优化、接触丰富控制**；Adams 服务 **汽车/航空/机械的系统级载荷、耐久、K&C、控制联仿**。选型时勿混为一谈。
- **数值思想可迁移：** 「约束 → DAE → 稀疏结构保持的隐式积分」仍是理解现代多体求解器的底色；机器人侧读 [Drake](./drake.md) 的严谨多体与 [MuJoCo](./mujoco.md) 的接触凸优化时，可用 ADAMS 史作对照坐标。

## 核心原理

### 方法主线（学术一手）

```mermaid
flowchart LR
  thesis["Orlandea 1973 论文<br/>node-analogous + STF + BDF + Lagrange"]
  p1["1977 ASME Part 1<br/>运动方程与约束"]
  p2["1977 ASME Part 2<br/>力元 / 静力·瞬态·线性化 / 模态优化"]
  prog["ADAMS 程序实现"]
  ex["算例：Malibu 悬架 · 747 起落架"]
  thesis --> p1 --> prog
  p1 --> p2 --> prog
  prog --> ex
```

1. **列式：** 将三维机构写成带约束的动力学系统；稀疏 tableau（STF）保留大系统稀疏性，使「先列式再数值」与稀疏求解一致。
2. **数值：** 借用电路仿真成熟的 **稀疏线性求解 + stiff 积分（Gear BDF）**，抑制宽谱特征值带来的不稳定。
3. **分析类型（Part 2）：** 静力、瞬态、线性化，以及模态相关优化；弹簧/阻尼等力函数进入同一程序框架。
4. **验证叙事：** 工业机构算例 + 与实验对照（摘要中强调 tabular efficiency / experimental comparison）。

### 商业化时间线（交叉核对）

| 节点 | 内容 |
|------|------|
| 1971 | Maros & Orlandea：多自由度运动方程、面向编程 |
| 1973 | Orlandea 博士论文；程序命名 ADAMS |
| 1976 | Mechanical Dynamics, Inc.（MDI）创办；Orlandea 为原初架构师 |
| 1977 | ASME Part 1 / Part 2 发表 ADAMS 方法与实现 |
| 其后 | MSC Software → Hexagon → **Cadence** 产品线；Adams Car、Adams Real Time 等 |

## 工程实践

| 场景 | 读法 |
|------|------|
| 需要 **整车 K&C / 路载 / 耐久 / NVH** | 看当代 [Cadence Adams 产品页](../../sources/sites/cadence-msc-adams.md)；依赖商业许可与 SimCompanion 文档 |
| 需要 **Simulink / ECU / HIL** | 产品页强调 Adams Controls 与 Simulink、Real-Time / DIL–HIL |
| 需要 **开源 RL 并行训练** | **不要选 Adams**；改 [MuJoCo](./mujoco.md) / [Isaac Lab](./isaac-lab.md) / [Genesis](./genesis-sim.md)（见 [仿真器选型](../queries/simulator-selection-guide.md)） |
| 需要 **严谨优化友好多体** | 优先 [Drake](./drake.md)；ADAMS 史作工业对照 |
| 复现学术配方 | 读 [一手论文索引](../../sources/papers/adams_orlandea_primary_refs.md)；学位论文 PDF 在 Deep Blue **校园限制** |

### 开源状态（2026-07-28 核查）

| 项 | 结论 |
|----|------|
| 方法论文 | **已公开发表**（ASME DOI；摘要可核） |
| 1973 学位论文 PDF | Deep Blue **校园限制**；DOI / Handle 可引 |
| 当代 Adams 求解器源码 | **确认未开源**（商业 CAE） |
| 源码运行时序图 | **不适用**（无可运行公开官方仓） |

## 局限与风险

- **许可与生态封闭：** 无法像 MuJoCo 一样 `pip`/克隆后做大规模策略采样；学术复现止于论文级理解或机构授权。
- **目标函数不同：** 工业「一次正确的系统级预测」≠ 机器人「百万环境步的吞吐」；把 Adams 当 RL backend 会选错栈。
- **全文获取：** 1977/2016 ASME 全文依赖订阅；勿仅凭二手博客转述公式细节。
- **命名混淆：** 产品常写作 **Adams**；全称 **Automatic Dynamic Analysis of Mechanical Systems** 才是学术程序名；与 unrelated 「ADAMS 优化算法」等缩写撞名时需看语境。

## 关联页面

- [MuJoCo](./mujoco.md) — 开源接触丰富刚体引擎；RL/控制研究默认对照
- [Drake](./drake.md) — 优化优先的严谨多体与系统框架
- [Motrix](./motrix.md) — 工业叙事下的现代高性能多体/训练平台（开源侧）
- [PyBullet](./pybullet.md) — 轻量入门物理绑定
- [十年仿真平台技术地图](../overview/sim-platforms-decade-technology-map.md) — 机器人学习侧平台史（ADAMS 属更早工业 CAE 层）
- [Locomotion RL 仿真器选型](../queries/simulator-selection-guide.md) — 明确「不要用工业 Adams 替代 RL 仿真器」
- [Sim2Real](../concepts/sim2real.md) — 仿真可信度与迁移；工业虚拟样机是另一条验证文化

## 参考来源

- [ADAMS Orlandea 一手学术索引](../../sources/papers/adams_orlandea_primary_refs.md) — 1971 / 1973 / 1977×2 / 2016
- [U-M Deep Blue 学位论文馆藏页](../../sources/sites/umich-deepblue-orlandea-adams-thesis.md)
- [Cadence MSC Adams 产品页](../../sources/sites/cadence-msc-adams.md)
- [Janevic 纪念文（商业化时间线交叉）](../../sources/blogs/janevic_orlandea_adams_memorial.md)

## 推荐继续阅读

- Orlandea, Chace, Calahan (1977) Part 1 & 2 — DOI [10.1115/1.3439312](https://doi.org/10.1115/1.3439312) · [10.1115/1.3439313](https://doi.org/10.1115/1.3439313)
- Orlandea (2016) *Multibody Systems History of ADAMS* — DOI [10.1115/1.4034296](https://doi.org/10.1115/1.4034296)
- Cadence Adams 产品页 — <https://www.cadence.com/en_US/home/tools/msc-software/adams.html>
