# Tesla 人形机器人膝关节机构专利（WO2024073135A1）

> 专利来源归档（ingest）

- **标题：** Systems and methods for a robot knee joint assembly
- **类型：** patent / humanoid / hardware / actuator
- **机构：** Tesla Inc.
- **链接：** <https://patents.google.com/patent/WO2024073135A1/en>
- **入库日期：** 2026-07-05
- **一句话说明：** 双连杆 + **线性执行器** 驱动的膝部 **四杆/双枢轴** 机构，使小腿相对大腿绕第二枢轴旋转，属于人形 **并联/连杆式关节模组** 专利布局。

## 核心摘录（面向 wiki 编译）

### 1) 膝关节连杆—线性执行器构型

- **要点：** **第一连杆** 一端铰接于大腿并绕 **第一枢轴** 转动；**第二连杆** 一端铰接小腿；**线性执行器** 连接两连杆远端，伸缩时驱动小腿绕 **第二枢轴** 相对大腿摆动——用 **直线电机/丝杠** 替代传统同轴旋转关节，利于 **力矩密度与封装** 权衡。
- **对 wiki 的映射：**
  - [`wiki/entities/patent-tesla-robot-knee-joint-assembly.md`](../../wiki/entities/patent-tesla-robot-knee-joint-assembly.md)
  - [`wiki/entities/humanoid-robot.md`](../../wiki/entities/humanoid-robot.md)

### 2) 与 Boston Dynamics 全电关节路线的对比语境

- **要点：** BD 专利 US10525601 强调 **旋转电机 + 关节内 PCB 驱动**；Tesla 本件强调 **连杆机构 + 线性执行器** 的 **非共轴膝部** 设计，反映人形硬件 **机构学路线分化**。
- **对 wiki 的映射：**
  - [`wiki/entities/patent-boston-dynamics-legged-control-stack.md`](../../wiki/entities/patent-boston-dynamics-legged-control-stack.md)

## 当前提炼状态

- [x] 摘要级摘录与 wiki 映射
- [ ] 待 Tesla Optimus 公开技术报告与更多同族专利交叉核对
