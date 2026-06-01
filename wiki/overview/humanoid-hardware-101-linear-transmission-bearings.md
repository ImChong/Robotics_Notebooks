---
type: overview
tags: [humanoid, hardware, ball-screw, roller-screw, bearing, category-hub]
status: complete
updated: 2026-06-01
summary: "Humanoid Hardware 101 · 03 直线传动与轴承 — 滚珠丝杠占 BOM ~20%、行星滚柱丝杠为膝踝髋瓶颈；交叉滚子轴承承弯矩。"
related:
  - ./humanoid-hardware-101-technology-map.md
  - ./humanoid-hardware-101-actuation-sensing-chain.md
  - ./humanoid-hardware-101-integrated-actuators.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_hardware_101.md
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
---

# Humanoid Hardware 101 · 03：直线传动与轴承

> **图谱分类节点**：**丝杠、轴承** 章；旋转电机→直线推力的「机械肌肉」，以及关节 **三向载荷** 支承。

## 丝杠

| 类型 | 机理 | 人形角色 | 瓶颈 |
|------|------|----------|------|
| **滚珠丝杠** | 滚珠滚动、效率高 | 肢体/躯干直线执行器 | 点接触，跳跃冲击易 **布氏压痕** |
| **行星滚柱丝杠** | 线接触分散载荷 | 膝踝髋高性能关节 | **供应链紧、成本高**（文内称行业显著瓶颈） |
| **梯形丝杠** | 滑动 | 低成本/低动态 | 效率与寿命受限 |

- 丝杠约占当前 BOM **~20%**；欧日主导磨削滚珠丝杠，中国 **轧制** 丝杠降本推动规模化兴趣。
- 滚珠丝杠腿关节需 **连杆+杆端**，系统复杂度高于旋转关节直驱方案。

## 轴承

- 深沟球、角接触、推力等按载荷选型；**交叉滚子轴承** 同时抗径向/轴向/弯矩，用于髋膝肩腕等核心关节。
- 手部微型高精度轴承难享大宗规模效应。

## 关联页面

- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [传动与感知链](./humanoid-hardware-101-actuation-sensing-chain.md)
- [集成执行器](./humanoid-hardware-101-integrated-actuators.md)
- [产业与成本地缘](./humanoid-hardware-101-supply-chain-economics.md)

## 参考来源

- [wechat_human_five_humanoid_hardware_101.md](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md)
- [wechat_humanoid_hardware_101_2026-06-01.md](../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)
