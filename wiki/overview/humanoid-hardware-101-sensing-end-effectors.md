---
type: overview
tags: [humanoid, hardware, tactile, end-effector, imu, category-hub]
status: complete
updated: 2026-06-01
summary: "Humanoid Hardware 101 · 06 传感与末端 — IMU/相机可commodity；触觉占手部 BOM 大头；多数任务不必 24DoF 全驱动灵巧手。"
related:
  - ./humanoid-hardware-101-technology-map.md
  - ./humanoid-hardware-101-integrated-actuators.md
  - ../concepts/visuo-tactile-fusion.md
  - ../methods/grasp-pose-estimation.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_hardware_101.md
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
---

# Humanoid Hardware 101 · 06：传感与末端执行器

> **图谱分类节点**：**通用传感器、触觉传感器、末端执行器** 三章。

## 通用传感器

- **IMU（MEMS）**、相机模组等受益于 **汽车/消费电子** 规模。
- 力矩/接触常可由 **电机电流、关节扭矩估计、足部接触** 间接获得，不必处处专用载荷传感器。
- 当前多 **冗余传感** 弥补标定与算法不成熟；感知成熟后可 **减传感器降 BOM**（文内引 Physical Intelligence 等「纯视觉+简单夹爪」案例）。

## 触觉

- 高精度灵巧手中，**触觉可占单手 BOM ~40%**（文内假设指尖/掌心有限覆盖、~60 传感单元量级）。
- 与 [集成执行器](./humanoid-hardware-101-integrated-actuators.md) 的空心杯、微型丝杠成本叠加。

## 末端执行器

- 全驱动多指 **成本高、装配难、耐久挑战大**；多数商业场景 **两指/三指/任务夹爪** 单位经济性更优。
- 文内：工业取放、家务（Sunday 三指等）证明 **不必人手形态** 才能干活。

## 关联页面

- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [产业与成本地缘](./humanoid-hardware-101-supply-chain-economics.md)
- [视觉触觉融合](../concepts/visuo-tactile-fusion.md)
- [抓取姿态估计](../methods/grasp-pose-estimation.md)

## 参考来源

- [wechat_human_five_humanoid_hardware_101.md](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md)
- [wechat_humanoid_hardware_101_2026-06-01.md](../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)
