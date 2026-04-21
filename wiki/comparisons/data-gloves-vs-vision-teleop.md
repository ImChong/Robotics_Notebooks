---
type: comparison
tags: [dexterity, data-collection, teleoperation, hardware, sensors]
status: complete
updated: 2026-04-21
related:
  - ../queries/dexterous-data-collection-guide.md
  - ../entities/allegro-hand.md
  - ../methods/behavior-cloning.md
sources:
  - ../../sources/papers/imitation_learning.md
summary: "灵巧操作数据采集选型：对比了穿戴式数据手套与基于视觉的遥操作方案，涵盖了精度、成本、遮挡鲁棒性及力反馈等关键维度。"
---

# 数据手套 vs 视觉遥操作 (灵巧数据采集选型)

在训练灵巧手（Dexterous Hand）执行复杂任务时，获取高质量的人类演示数据是第一步。目前，**穿戴式数据手套 (Data Gloves)** 和 **基于视觉的遥操作 (Vision-based Teleop)** 是两条最主流的技术路线。

## 核心对比

| 维度 | 穿戴式数据手套 | 基于视觉的遥操作 (如 Meta Quest/Leap) |
|------|--------------|---------------------------------|
| **测量原理** | 物理接触（弯曲传感器/IMU） | 非接触式光电（RGB/深度相机） |
| **精度与稳定性** | ✅ 极高且稳定，无漂移 | ⚠️ 中等，受光照和背景干扰，存在漂移 |
| **遮挡鲁棒性** | ✅ 免疫遮挡，手放在背后也能测 | ❌ 极差，手指交叠或被物体遮挡时失效 |
| **力反馈 (Haptics)** | ✅ 部分高级型号支持主动力反馈 | ❌ 几乎没有，只能靠视觉观察 |
| **操作自由度** | ⚠️ 受限，手部有异物感，容易疲劳 | ✅ 极高，裸手操作，自然丝滑 |
| **成本** | ❌ 昂贵 (5k - 50k USD) | ✅ 廉价 (100 - 500 USD) |
| **校准复杂度** | 繁琐，每个人的手部尺寸需精确适配 | 简单，通过算法自动重定向 |

## 适用场景分析

### 推荐使用数据手套的场景：
1. **接触极其频繁的操作**：例如在口袋里摸索硬币、盲拧螺丝。这类任务视觉无法介入，且需要手套提供极高频的关节序列。
2. **高保真力控训练**：当你的 RL 奖励函数对手指捏力非常敏感时，带有力反馈的数据手套（如 SenseGlove）是唯一的选择。
3. **长期、大规模工程采集**：对数据一致性（Consistency）要求极高，不能容忍由于摄像头位置变动导致的轨迹偏移。

### 推荐使用视觉遥操作的场景：
1. **低预算科研项目**：Meta Quest 配合重定向算法（Retargeting）可以快速搭建起一套可用的 Demo 采集系统。
2. **需要极致自然动作的模仿**：人类在裸手状态下展现出的灵巧度远超戴着沉重手套的状态，采集到的轨迹更符合“专家”定义。
3. **视觉引导的常规操作**：如折衣服、整理桌面、倒水。这类任务遮挡时间短，算法可以通过时序滤波进行平滑。

## 融合趋势：视触觉混合采集

目前顶尖的实验室（如斯坦福 ALOHA 团队）开始探索将两者结合：使用 Quest 进行宏观位姿追踪，而在指尖安装 GelSight 传感器采集微观接触特征，从而兼顾成本、灵巧度与数据精度。

## 关联页面
- [灵巧操作数据采集指南](../queries/dexterous-data-collection-guide.md)
- [Allegro Hand 实体](../entities/allegro-hand.md)
- [Behavior Cloning (行为克隆)](../methods/behavior-cloning.md)

## 参考来源
- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md)
- Qin, B., et al. (2023). *AnyTeleop Framework*.
