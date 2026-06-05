---
type: overview
tags: [humanoid, hardware, bom, supply-chain, china, usa, category-hub]
status: complete
updated: 2026-06-01
summary: "Humanoid Hardware 101 · 07 产业与成本地缘 — 执行器难 10× 降本；传感器 3–5× 路径；中国靠 EV/无人机/消费电子供应链+产业密度，美国资本偏软件等待操作智能成熟。"
related:
  - ./humanoid-hardware-101-technology-map.md
  - ./humanoid-hardware-101-integrated-actuators.md
  - ./humanoid-hardware-101-sensing-end-effectors.md
  - ../queries/humanoid-hardware-selection.md
  - ../entities/open-source-humanoid-hardware.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_hardware_101.md
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
---

# Humanoid Hardware 101 · 07：产业与成本地缘

> **图谱分类节点**：**人形机器人产业格局、成本分析、地缘格局、潜在领先者** 及文末观察。

## BOM 结构（文内量级）

- **执行器**：整机最大头；宇树类 **~2000 元/关节×25–30**；减速器占执行器 **45–50%**，电机 **20–25%**，编码器 **10–15%**。
- **丝杠**：整机 BOM **~20%**。
- **手部**：空心杯 ~30%、触觉 ~40%、减速/丝杠 ~20%、结构 ~10%（文内单手估算）。
- **2 万美元目标**：需 **减关节、共享模块、简化手**；供应商压价仅 **2–3×** 边际，非 10×。

## 分部件降本潜力（策展）

| 子系统 | 文内判断 |
|--------|----------|
| 执行器/关节 | 降本斜率最陡约束在 **机械耦合**；趋势反而 **执行器数量增加** |
| 传感器 | **3–5×** 路径，冗余可随算法减少 |
| 计算/PCB | BOM 占比低，DFM/复用有效；算力芯片短期难大跌 |
| 电池 | 受益 EV 链，增量在封装与 BMS |
| 结构件 | 铸造、注塑、部件整合等 **传统制造曲线** |

## 中美三维框架

| 维度 | 中国 | 美国 |
|------|------|------|
| **供应链** | EV+无人机+消费电子→四大件可组装；端到端密度 | 等待操作智能；大厂投资多来自企业战略而非纯 VC |
| **需求** | 政府采购、场景营销、循环采购等 **创造需求** | 劳动力贵但需机器人 **不改流程即可用** |
| **资本** | 份额+产能优先，省级产业竞争 | 硬件 COGS 不确定抑制过早 mega-factory |

## 开放问题（文内）

- 价值捕获在整机、部署方、部件商还是组合？**尚无明确全球领先者**。
- **混合模式**（美研软件+中制硬件）、关税下 **泰国/墨西哥组装**、中国研究 **逼近前沿** 等均为 2026 观察点。

## 关联页面

- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [人形硬件选型 Query](../queries/humanoid-hardware-selection.md)
- [集成执行器](./humanoid-hardware-101-integrated-actuators.md)
- [传感与末端](./humanoid-hardware-101-sensing-end-effectors.md)

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |
| PCB | Printed Circuit Board | 印刷电路板 |
| DFM | Design for Manufacturing | 面向制造的设计，降低量产成本与风险 |
| BMS | Battery Management System | 电池管理系统 |

## 参考来源

- [wechat_human_five_humanoid_hardware_101.md](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md)
- [wechat_humanoid_hardware_101_2026-06-01.md](../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)
