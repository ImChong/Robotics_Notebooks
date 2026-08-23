---
type: concept
tags: [humanoid, hardware, manufacturing, dfm, supply-chain, reliability, yield]
status: complete
updated: 2026-08-23
related:
  - ./embodied-foundation-model-hardware-codesign.md
  - ../overview/humanoid-hardware-101-supply-chain-economics.md
  - ../overview/humanoid-hardware-101-actuation-sensing-chain.md
  - ./humanoid-knee-harmonic-drive-limits.md
  - ./planetary-roller-screw-humanoid-leg-actuation.md
  - ../overview/humanoid-hardware-101-technology-map.md
  - ../queries/humanoid-hardware-selection.md
  - ../entities/unitree-g1.md
  - ../roadmaps/humanoid-practitioner-entry-roadmap.md
sources:
  - ../../sources/blogs/wechat_zanehub_humanoid_mass_production_experience.md
summary: "人形量产经验是把技术可行性转化为制造可行性的系统工程能力：三大核心件（谐波/PRS/无框电机）工艺定型、良率 S 曲线与 CPK 门槛、供应链一致性与 ISO/IEC 可靠性体系，并可从汽车 PPAP 与 3C MES 等跨行业迁移——样机与资本不等于量产。"
---

# 人形机器人量产工程能力

## 一句话定义

**人形机器人量产经验**不是「做出能走路的样机」，而是把关节级 **工艺 know-how、良率爬坡、供应链一致性与可靠性验证** 固化成可重复放大产线的 **制造可行性**——核心件（谐波减速器、行星滚柱丝杠、无框力矩电机）的 DFM 定型往往比算法 demo 更决定能否从千台走向万台。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DFM | Design for Manufacturing | 面向制造的设计，设计阶段引入工艺约束 |
| PPAP | Production Part Approval Process | 生产件批准程序，汽车行业批量放行流程 |
| CT | Cycle Time | 产线节拍，单件完成时间 |
| FPY | First Pass Yield | 首件合格率，首次通过无需返工的比例 |
| CPK | Process Capability Index | 过程能力指数，衡量尺寸/性能波动 |
| AQL | Acceptable Quality Level | 可接受质量水平，批产抽检口径 |
| PRS | Planetary Roller Screw | 行星滚柱丝杠，直线关节常见传动 |
| PFMEA | Process Failure Mode and Effects Analysis | 过程失效模式与影响分析 |

## 为什么重要

- **2026「量产元年」叙事下**，公开报道里 Optimus、Walker、IRON、宇树等产能目标与单价带宽并存——读者需要区分 **媒体样机** 与 **S 曲线良率、CPK 与 PPAP 放行** 两套语言。
- **与 [Hardware 101 · 产业成本](../overview/humanoid-hardware-101-supply-chain-economics.md) 互补**：该页回答 BOM 与地缘；本页回答 **如何把四大件造稳、造一致**。
- **给选型留制造侧判据**：谐波不是只看额定扭矩——[膝侧避开谐波](./humanoid-knee-harmonic-drive-limits.md) 谈冲击谱载与柔轮疲劳；量产侧还要问 **几十万件的良率与材料批次一致性**。
- **与大模型叙事衔接**：自研本体不仅是制造问题，更是 **数据—仿真—量产同线** 的系统定义权；见 [具身大模型与本体协同设计](./embodied-foundation-model-hardware-codesign.md)。

## 核心原理

### 量产经验 = 技术可行性 → 制造可行性

行业常分三阶段：**科研样机 → 工业/商业应用 → 产业化**。真正值钱的不是单一技术突破或资本投入，而是：

1. **硬技术门槛**：核心件工艺定型、良率爬坡、按 ISO/IEC 体系的可靠性验证  
2. **软管理门槛**：DFM 评审、供应链管理与跨行业经验迁移  
3. **隐性资产**：工艺参数库、失效模式库、从 90% 爬到 99% 的方法论  

### 流程总览：从 DFM 到放行

```mermaid
flowchart LR
  DFM[DFM 评审<br/>设计冻结前引入制造意见]
  Proc[三大核心件<br/>工艺定型]
  Ramp[良率 S 曲线爬坡<br/>中试→小批→规模]
  SC[供应链一致性<br/>材料/绕线/磨削]
  Rel[可靠性验证<br/>ISO/IEC + 寿命台架]
  PPAP[PPAP / 控制计划<br/>CPK≥1.33 等]
  Ship[规模交付]
  DFM --> Proc --> Ramp --> SC --> Rel --> PPAP --> Ship
```

### 三大核心件工艺命门

| 部件 | 在整机中的位置 | 工艺命门 | 量产侧关键指标 |
|------|----------------|----------|----------------|
| **谐波减速器** | 旋转关节，约占整机成本 15–20% | 柔轮薄壁 **材料+热处理**（软以便反复弯曲、硬以防开裂）；传动精度到弧秒级 | 疲劳寿命一致性、**批次良率**（文内公开案例：新一代产品良率约 92%） |
| **行星滚柱丝杠** | 直线/膝踝等关节 | **螺纹磨削**、滚柱尺寸一致性、装配同轴度 | 导程精度 G1–G5（国内试制常从 G5 起步）；见 [PRS 腿执行器](./planetary-roller-screw-humanoid-leg-actuation.md) |
| **无框力矩电机** | 关节近端驱动 | **绕线张力**、磁钢批次一致性、大力矩 **散热** | 电气参数与扭矩常数批次波动；自动绕线/浸漆参数固化 |

### 良率、节拍与 CPK

**典型良率 S 曲线（工程量级，非厂商承诺）：**

| 阶段 | 良率区间 | 含义 |
|------|----------|------|
| 手工样件 | 30–50% | 工艺验证 |
| 中试线 | 60–70% | 工艺固化 |
| 小批量 | 80–85% | 工艺优化 |
| 规模量产 | 90–95% | 工艺稳定 |
| 成熟量产 | ≥98% | 工艺成熟 |

从 92% 爬到 98–99% 的 **边际工程代价** 往往远大于 60%→90%——这是「龙头供应商」与「能做样机」的分水岭。

**产线节拍 CT（文内典型量级）：** 关节模组装配 15–30 min/件；减速器测试 5–10 min；电机绕线 3–5 min。  
**FPY：** 首批 ≥95% 一次合格。  
**CPK：** 关键尺寸 ≥1.33（≈99.99%）常被视为进主流供应链的 **过程能力门槛**。

### 跨行业可迁移经验（摘要）

| 来源 | 可直接借鉴 | 需重新签核 |
|------|------------|------------|
| **汽车** | DFM、PPAP（PFMEA/控制计划/MSA/初始 CPK）、产线 IE | ISO 26262 与体积重量约束 |
| **3C** | 光学校准、SMT、AOI、MES 追溯 | 产品生命周期与精度诉求 |
| **医疗器械** | 生物相容性测试思路、高精度加工 | 监管体系 |
| **航天** | FMEA、冗余、HALT | 成本与重量 |
| **手机精密制造** | CNC、阳极氧化、喷涂 | 结构件载荷与材料 |

## 工程实践

### 可靠性测试主干（选型/验收清单）

- **标准族：** ISO 9283（性能）、ISO 10218/13849（安全）、IEC 61508（功能安全）、IEC 60068-2 / MIL-STD-810H（环境）、GB/T 38559 等。  
- **环境：** 85°C/85%RH 存储、-40~+85°C 温度循环、盐雾、多轴振动。  
- **寿命台架：** 关节模组周级循环、减速器万小时级、电机绝缘寿命；配合 **扭转载荷谱** 与加速老化。  
- **批产：** AQL 0.65 + GB/T 2828.1 水平 II 等——与 CPK 互补，一个管 **过程**，一个管 **批次放行**。

### 主流路径对照（公开叙事，非冻结 BOM）

| 路径 | 代表叙事 | 制造侧读法 |
|------|----------|------------|
| **垂直整合从零建线** | Tesla Optimus：大量独特零件、自研核心件、目标压到约 2 万美元/台 | 工艺验证与供应链 **同时** 从零开始，良率爬坡最长 |
| **渐进式工业场景** | 优必选 Walker：模块化、工业场景先行、单价带宽下行 | 先固化 **场景可交付** 再扩产能 |
| **车企制造迁移** | 小鹏 IRON：汽车级 DFM/产线经验 + VLA | PPAP 思维可迁移，但 **关节耦合公差链** 需重算 |
| **快速迭代小批量** | 宇树：研发当年小批量发货 | 先验证 **市场与软件栈**，规模一致性是下一关 |

### 评估供应商时至少追问

1. 当前处于良率 S 曲线哪一段？有无 **CPK 与 FPY** 数据？  
2. 核心件是 **自研、独家代工还是多源**？材料/磁钢/钢材批次一致性如何？  
3. 可靠性报告覆盖哪些 **标准与载荷谱**？是否只有额定扭矩台架而无步态冲击谱？  
4. PPAP/控制计划是否对客户开放？失效模式库是否可审计？

## 局限与风险

- **第三方解读边界**：本文编译自公众号工程叙事；产能、良率、投资金额等多来自公开报道，**会随财报与产品迭代变化**。  
- **不要把汽车 PPAP 生搬硬套**：人形关节数多、耦合公差链长，且 RL 时代存在 **可消耗件与 QDD** 路线，安全与寿命指标需按整机重新定义。  
- **样机 ≠ 量产**：媒体展示的行走/抓取不等于 CPK≥1.33 与万小时寿命；[硬件选型 Query](../queries/humanoid-hardware-selection.md) 应同时看 **科研平台** 与 **工业交付** 两条线。  
- **地缘与产能叙事**：[产业与成本](../overview/humanoid-hardware-101-supply-chain-economics.md) 中的中美供应链框架仍约束 **谁能先拿到稳定四大件**。

## 关联页面

- [Humanoid Hardware 101 · 产业与成本](../overview/humanoid-hardware-101-supply-chain-economics.md) — BOM、DFM 与地缘  
- [Humanoid Hardware 101 · 传动与感知链](../overview/humanoid-hardware-101-actuation-sensing-chain.md) — 谐波/电机/编码器部件层  
- [膝侧避开谐波工程判据](./humanoid-knee-harmonic-drive-limits.md) — 柔轮疲劳与冲击谱载  
- [PRS 腿执行器](./planetary-roller-screw-humanoid-leg-actuation.md) — 丝杠精度与线性关节  
- [Humanoid Hardware 101 技术地图](../overview/humanoid-hardware-101-technology-map.md) — 七类子系统 hub  
- [Unitree G1](../entities/unitree-g1.md) — 量产科研平台对照  

## 参考来源

- [wechat_zanehub_humanoid_mass_production_experience.md（仓库内归档）](../../sources/blogs/wechat_zanehub_humanoid_mass_production_experience.md) — Zane Hub 公众号：<https://mp.weixin.qq.com/s/CARW0vvd4doO1htt0Q1bHg>

## 推荐继续阅读

- [ISO 9283 工业机器人性能规范](https://www.iso.org/standard/63798.html) — 性能与校准测试基线  
- [Humanoid Hardware 101 原始长文编译](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md) — 部件级 BOM 与供应链背景  
