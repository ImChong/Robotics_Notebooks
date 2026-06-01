---
type: overview
tags: [humanoid, hardware, battery, compute, pcb, bms, category-hub]
status: complete
updated: 2026-06-01
summary: "Humanoid Hardware 101 · 05 能源与计算电子 — 锂电 NMC 标配、功率密度与能量密度难兼得；Thor/NX 主导算力；PCB/BMS 可随 DFM 复用降本。"
related:
  - ./humanoid-hardware-101-technology-map.md
  - ../queries/humanoid-battery-thermal-management.md
  - ../entities/open-source-humanoid-brains.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_hardware_101.md
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
---

# Humanoid Hardware 101 · 05：能源与计算电子

> **图谱分类节点**：**电池、计算单元、PCB** 三章 — 移动人形的 **能量预算** 与 **边缘算力**。

## 电池

- **锂离子** 标配（轻、高电压）；正极常见 **NMC** 等镍基体系利能量密度。
- 人形需同时：**高功率密度**（失稳毫秒级修正）+ **高能量密度**（整班续航）——化学体系 **难以双极致**。
- 电池包增重 → 更大电机扭矩 → 更高功耗（**重量螺旋**）；优化在 **标准电芯 + 封装集成 + 认证 BMS**。

## 计算单元

- 文内：**NVIDIA Thor / NX / ADX** 等在机器人边缘算力占主导，短期难大幅降价。
- 早期定制板卡贵，**模块复用 + DFM** 可快速降本（非硅价 alone）。

## PCB

- 驱动板、控制板、通信与电源管理；受益于 **面板化批量** 与汽车/移动电子生态。

## 关联页面

- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [人形电池热管理 Query](../queries/humanoid-battery-thermal-management.md)
- [开源人形「大脑」对比](../entities/open-source-humanoid-brains.md)
- [产业与成本地缘](./humanoid-hardware-101-supply-chain-economics.md)

## 参考来源

- [wechat_human_five_humanoid_hardware_101.md](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md)
- [wechat_humanoid_hardware_101_2026-06-01.md](../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)
