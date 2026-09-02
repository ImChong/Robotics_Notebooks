---
type: overview
tags: [humanoid, hardware, tactile, end-effector, imu, category-hub]
status: complete
updated: 2026-09-02
summary: "Humanoid Hardware 101 · 06 传感与末端 — IMU/相机可commodity；触觉占手部 BOM 大头；多数任务不必 24DoF 全驱动灵巧手。"
related:
  - ./humanoid-hardware-101-technology-map.md
  - ./humanoid-hardware-101-integrated-actuators.md
  - ../concepts/visuo-tactile-fusion.md
  - ../methods/grasp-pose-estimation.md
  - ../entities/sunday-robotics-act2.md
  - ../entities/xyz-deux.md
  - ../entities/twindex.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_hardware_101.md
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
  - ../../sources/sites/xyzcorp-deux.md
  - ../../sources/sites/x2robot-twindex.md
---

# Humanoid Hardware 101 · 06：传感与末端执行器

> **图谱分类节点**：**通用传感器、触觉传感器、末端执行器** 三章。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |

## 通用传感器

- **IMU（MEMS）**、相机模组等受益于 **汽车/消费电子** 规模。
- 力矩/接触常可由 **电机电流、关节扭矩估计、足部接触** 间接获得，不必处处专用载荷传感器。
- 当前多 **冗余传感** 弥补标定与算法不成熟；感知成熟后可 **减传感器降 BOM**（文内引 Physical Intelligence 等「纯视觉+简单夹爪」案例）。

## 触觉

- 高精度灵巧手中，**触觉可占单手 BOM ~40%**（文内假设指尖/掌心有限覆盖、~60 传感单元量级）。
- 与 [集成执行器](./humanoid-hardware-101-integrated-actuators.md) 的空心杯、微型丝杠成本叠加。

## 末端执行器

- 全驱动多指 **成本高、装配难、耐久挑战大**；多数商业场景 **两指/三指/任务夹爪** 单位经济性更优。
- 文内：工业取放、家务（[Sunday 三指](../entities/sunday-robotics-act2.md) 等）证明 **不必人手形态** 才能干活；韩国 [DEUX](../entities/xyz-deux.md)（XYZ，2026）同样押注 **三指专有手 + Glove X 1:1 采数**；自变量 [TwinDEX](../entities/twindex.md)（2026-09）把三指定为「灵巧最小可行解」并用 **7 主动 / 2 被动** 外骨骼–机械手共设计做无本体采数。

## 关联页面

- [ACT-2 / Sunday Robotics](../entities/sunday-robotics-act2.md) — Memo 家用平台与三指家务夹爪产业案例
- [DEUX（XYZ）](../entities/xyz-deux.md) — 半人形服务机器人 + 三指手 + Glove X 商业样本
- [TwinDEX](../entities/twindex.md) — 三指外骨骼–同构手共设计；无本体采数（自变量，闭源）
- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [产业与成本地缘](./humanoid-hardware-101-supply-chain-economics.md)
- [视觉触觉融合](../concepts/visuo-tactile-fusion.md)
- [抓取姿态估计](../methods/grasp-pose-estimation.md)

## 参考来源

- [wechat_human_five_humanoid_hardware_101.md](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md)
- [wechat_humanoid_hardware_101_2026-06-01.md](../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)
- [xyzcorp-deux.md](../../sources/sites/xyzcorp-deux.md) — DEUX 三指手与 Glove X 产品归档
- [x2robot-twindex.md](../../sources/sites/x2robot-twindex.md) — TwinDEX 三指外骨骼共设计归档
