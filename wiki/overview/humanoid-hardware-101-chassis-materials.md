---
type: overview
tags: [humanoid, hardware, chassis, materials, category-hub]
status: complete
updated: 2026-06-01
summary: "Humanoid Hardware 101 · 01 机身与材料 — 按载荷路径选材：铝合金承力骨架、钢耐磨、镁/钛减重、复材肢段、高分子外壳。"
related:
  - ./humanoid-hardware-101-technology-map.md
  - ./humanoid-hardware-101-integrated-actuators.md
  - ../queries/humanoid-hardware-selection.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_hardware_101.md
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
---

# Humanoid Hardware 101 · 01：机身与材料

> **图谱分类节点**：对应 [human five · Hardware 101](https://mp.weixin.qq.com/s/10hYwFzC1EuCypFVzC6QGQ) **机身骨架** 章；总地图见 [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GF | Glass Fiber | 玻璃纤维增强材料 |
| POM | Polyoxymethylene | 聚甲醛工程塑料（俗称赛钢/Delrin） |

## 核心问题

承重结构应按 **载荷路径** 分功能选材，而非把机身当作单一整体。

| 功能类 | 典型部件 | 材料倾向 |
|--------|----------|----------|
| 梁/柱传力 | 骨盆、躯干、连杆 | **铝合金**（比强度、可加工肋板/空心结构） |
| 高局部应力 | 电机座、齿轮箱壳、轴承座 | **钢**（耐磨、抗交变疲劳） |
| 极致减重 | 肢段、关键支架 | **镁合金**（轻 ~35% vs 铝，消防与供应链集中在中国）、**钛**（3D 打印友好、加工难） |
| 长肢段刚度 | 大腿、小腿 | **碳纤维/玻纤**（各向异性、与金属连接电偶腐蚀） |
| 非承力 | 外壳、线束导向 | **工程塑料**（GF 尼龙、POM 等） |

## 设计权衡（文内）

- 铝合金长期疲劳需 **圆角、钢衬套螺纹孔**；钢用于齿轮/轴/轴承等 **数百万次微动** 部位。
- 镁：**触变注射成型** 可降低液态燃烧风险；全球产能高度集中中国。
- 复材与金属连接是 **薄弱节点**；多用于对轻量化极敏感的长肢段。

## 关联页面

- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [集成执行器](./humanoid-hardware-101-integrated-actuators.md)
- [产业与成本地缘](./humanoid-hardware-101-supply-chain-economics.md)

## 参考来源

- [wechat_human_five_humanoid_hardware_101.md](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md)
- [wechat_humanoid_hardware_101_2026-06-01.md](../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)
