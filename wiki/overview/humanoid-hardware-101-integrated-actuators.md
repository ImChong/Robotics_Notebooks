---
type: overview
tags: [humanoid, hardware, actuator, electric, hydraulic, category-hub]
status: complete
updated: 2026-06-01
summary: "Humanoid Hardware 101 · 04 集成执行器 — 电动主导（线束+软件）；腿膝踝偏直线高力、肩等大转角用旋转；Atlas 由液压转电、部分灵巧手仍液压。"
related:
  - ./humanoid-hardware-101-technology-map.md
  - ./humanoid-hardware-101-actuation-sensing-chain.md
  - ./humanoid-hardware-101-linear-transmission-bearings.md
  - ../entities/open-source-humanoid-hardware.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_hardware_101.md
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
---

# Humanoid Hardware 101 · 04：集成执行器

> **图谱分类节点**：**执行器** 章 — 电机+减速+传感+驱动的一体化 **关节模块**（如擎天柱文内 ~28 个执行器量级）。

## 核心结论

- **电动** 主导：布线简单、与传感/软件天然兼容；液压仍胜 **紧凑极大力**（部分科研手/腿），气动适合 **二元/柔顺** 动作。
- **直线执行器**：腿 **膝踝** 等需高力、抗冲击时常用滚珠/滚柱丝杠直线方案。
- **旋转执行器**：肩等 **大转角** 关节更常见。

## 动力源对比（文内表意）

| 指标 | 电动 | 液压 | 气动 |
|------|------|------|------|
| 功率密度 | 中–高 | 最高 | 低–中 |
| 可控性 | 优 | 良 | 差 |
| 效率 | ~90% | 30–60% | 极低 |
| 系统复杂度 | 低（电线） | 高（泵管路） | 中 |

## 历史注记

- **Atlas** 液压运行多年后新一代转电；Sanctuary 手、Clone 等仍探索 **微液压/气动**  niche。
- 选型需平衡：功率密度、可控性、反向驱动、效率、响应、**系统复杂度**。

## 关联页面

- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [传动与感知链](./humanoid-hardware-101-actuation-sensing-chain.md)
- [传感与末端执行器](./humanoid-hardware-101-sensing-end-effectors.md)
- [产业与成本地缘](./humanoid-hardware-101-supply-chain-economics.md)

## 参考来源

- [wechat_human_five_humanoid_hardware_101.md](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md)
- [wechat_humanoid_hardware_101_2026-06-01.md](../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)
