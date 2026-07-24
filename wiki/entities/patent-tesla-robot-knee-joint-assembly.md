---
type: entity
tags:
  - humanoid
  - patent
  - hardware
  - actuator
status: complete
updated: 2026-07-05
related:
  - ./humanoid-robot.md
  - ./boston-dynamics.md
  - ./patent-boston-dynamics-legged-control-stack.md
  - ../concepts/humanoid-parallel-joint-kinematics.md
  - ../queries/actuator-drive-chain-selection-loop.md
sources:
  - ../../sources/patents/tesla_robot_knee_joint_wo2024073135.md
summary: "Tesla WO2024073135：双连杆 + 线性执行器的人形膝关节四杆机构，小腿绕大腿第二枢轴摆动，属连杆式关节模组专利。"
---

# Tesla 人形机器人膝关节机构（WO2024073135）

专利 **WO2024073135A1**（*Systems and methods for a robot knee joint assembly*，权利人 **Tesla Inc.**）公开一种 **双连杆 + 线性执行器** 的 **膝关节总成**：通过连杆机构将 **直线驱动** 转化为小腿相对大腿的 **旋转运动**，属于人形 **非共轴关节模组** 路线。

## 一句话定义

**用两根连杆把线性执行器的伸缩转成膝部弯曲，而不是在膝轴上直接堆同轴旋转电机。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WO | WIPO Patent Application | 国际专利申请公开 |
| IK | Inverse Kinematics | 由足端/关节角目标反求连杆驱动量 |
| FK | Forward Kinematics | 由驱动量正算关节角 |
| BD | Boston Dynamics | 对比其旋转电机关节内集成路线 |
| Locomotion | Robot Locomotion | 膝部为步行力矩与姿态关键关节 |

## 为什么重要

- **硬件路线信号：** 与 [BD 关节内旋转电机 + 驱动 PCB](./patent-boston-dynamics-legged-control-stack.md) 不同，Tesla 选择 **连杆—直线驱动** 构型，反映人形 **力矩密度、封装与惯量** 的另一种工程取舍。
- **机构学入口：** 膝部 **四杆/双枢轴** 设计直接影响 **控制雅可比、背隙与冲击**，与 [人形并联关节运动学](../concepts/humanoid-parallel-joint-kinematics.md) 主题相邻。

## 核心结构

| 构件 | 作用 |
|------|------|
| **第一连杆** | 一端铰接 **大腿**，绕 **第一枢轴** 相对大腿转动 |
| **第二连杆** | 一端铰接 **小腿**；小腿整体可绕 **第二枢轴** 相对大腿摆动 |
| **线性执行器** | 连接两连杆远端；伸缩时驱动 **膝屈曲/伸展** |

## 常见误区或局限

- **误区：「专利图即 Optimus 量产膝」。** WO 公开文本为 **申请披露**，量产机构可能迭代。
- **局限：** 摘要级公开 **未给出** 力矩曲线、减速比与控制频率；需等待产品拆解或后续同族专利。

## 关联页面

- [人形机器人](./humanoid-robot.md)
- [Boston Dynamics](./boston-dynamics.md)
- [BD 足式控制专利栈](./patent-boston-dynamics-legged-control-stack.md)
- [人形并联关节运动学](../concepts/humanoid-parallel-joint-kinematics.md)
- [执行器驱动链选型闭环知识链](../queries/actuator-drive-chain-selection-loop.md) — 膝关节执行器总成专利是①层执行器机械设计的工业参照

## 参考来源

- [Tesla 膝关节专利摘录（WO2024073135）](../../sources/patents/tesla_robot_knee_joint_wo2024073135.md)

## 推荐继续阅读

- Google Patents：<https://patents.google.com/patent/WO2024073135A1/en>
- Tesla Optimus 公开演示（产品语境，非专利等同）
