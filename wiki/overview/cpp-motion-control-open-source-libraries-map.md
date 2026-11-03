---
type: overview
tags: [overview, motion-control, cpp, open-source, trajectory-generation, ethercat, ros2, wechat]
status: complete
updated: 2026-11-03
related:
  - ../entities/ruckig.md
  - ../entities/plcopen-pyplcopen.md
  - ../entities/fastaccelstepper.md
  - ../concepts/whole-body-control.md
  - ../concepts/ethercat-protocol.md
  - ../methods/model-predictive-control.md
sources:
  - ../../sources/blogs/wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md
summary: "Qt历险记 C/C++ 运动控制开源库清单的阅读坐标：14 个有仓库 / 官方文档的项目逐个回到一手资料核对（LICENSE / README / 官方文档 / 论文），新建 7 / 复用 7 个节点，并列出文章与一手资料不一致之处（Ruckig 许可、pyplcopen 许可与名称、SOEM 双许可等）。"
---

# C/C++ 运动控制开源库地图：轨迹生成、动力学、工业总线与 ROS 栈（一手资料核对版）

> **本页定位**：为 [C/C++ 运动控制开源库分类整理](https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw)（Qt历险记，2026-11-02）提供一手资料核对后的阅读坐标。文章是转述式清单、没有给出处，本页对每个项目都回到仓库 LICENSE / README 或官方文档核对；**以一手资料为准，文章说法仅作对照**。

## 一句话观点

**这份清单能用，但「许可」和「实时 / 可用性」两列不能直接信：Ruckig 与 pyplcopen 的许可写错，Ruckig 社区版带中间路点不是实时的，pyplcopen 的 EtherCAT 尚未完成，SOEM 商用需商业许可。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OTG | Online Trajectory Generation | 在线轨迹生成 |
| PLCopen | PLCopen motion control standard | 运动控制功能块标准 |
| DDP | Differential Dynamic Programming | 微分动态规划 |
| MPC | Model Predictive Control | 模型预测控制 |
| FOC | Field-Oriented Control | 磁场定向控制 |

## 为什么单独做这张地图

- 文章把许可当「选型快照」给出，但其中有一手资料不支持的说法；运动控制库的许可（尤其 GPL / 双许可）与「社区版 vs Pro 版」直接影响能否商用与能否硬实时。
- 全库已有 7 个相关节点分散在各处，新增 7 个后需要统一入口。

## 项目一览（许可为一手核对结果）

| 项目 | 类别 | 节点 | 状态 | 许可（核查） |
|------|------|------|------|--------------|
| Pinocchio | 机器人动力学 / 整机控制库 | [pinocchio](../entities/pinocchio.md) | 复用 | BSD-2-Clause |
| Crocoddyl | 机器人动力学 / 整机控制库 | [crocoddyl](../entities/crocoddyl.md) | 复用 | BSD-3-Clause |
| SOEM | 工业总线与控制栈 | [soem](../entities/soem.md) | 复用 | GPLv3 + 商业双许可 |
| IgH EtherCAT Master | 工业总线与控制栈 | [igh-ethercat-master](../entities/igh-ethercat-master.md) | 复用 | GPL v2 |
| SimpleFOC（Arduino-FOC） | 工业总线与控制栈 | [simplefoc](../entities/simplefoc.md) | 复用 | MIT |
| ros2_control | ROS / ROS2 运动控制栈 | [ros2-control](../entities/ros2-control.md) | 复用 | Apache-2.0 |
| MoveIt 2 | ROS / ROS2 运动控制栈 | [moveit2](../entities/moveit2.md) | 复用 | BSD-3-Clause |
| Ruckig | 轨迹规划库 | [ruckig](../entities/ruckig.md) | 新建 | MIT |
| pyplcopen / plcopen 运动控制内核 | 轨迹规划库 | [plcopen-pyplcopen](../entities/plcopen-pyplcopen.md) | 新建 | Apache-2.0 |
| FastAccelStepper | 轨迹规划库 | [fastaccelstepper](../entities/fastaccelstepper.md) | 新建 | MIT |
| Robotics Library（RL） | 机器人动力学 / 整机控制库 | [robotics-library-rl](../entities/robotics-library-rl.md) | 新建 | BSD-2-Clause |
| Control Toolbox（ct） | 机器人动力学 / 整机控制库 | [control-toolbox-ct](../entities/control-toolbox-ct.md) | 新建 | BSD-2-Clause |
| RTmotion（Intel ECI） | 工业总线与控制栈 | [rtmotion-intel-eci](../entities/rtmotion-intel-eci.md) | 新建 | 文章写「开源」；本次未能定位公开仓库，许可未核实 |
| OMPL（Open Motion Planning Library） | ROS / ROS2 运动控制栈 | [ompl](../entities/ompl.md) | 新建 | BSD |

> 「WBC Whole-Body Control」文章只给通用名、无仓库，不能对应唯一项目，请见概念页 [全身控制](../concepts/whole-body-control.md)。

## 文章选型建议与核对

| 场景 | 文章建议 | 核对 |
|------|----------|------|
| 自研 EtherCAT 工业运动控制器（CNC / 机械手） | pyplcopen（plcopen 内核）+ SOEM / IgH + Ruckig（可选） | plcopen README 写 EtherCAT 未开工，需自行接主站；SOEM 商用需商业许可；Ruckig 社区版中间路点非实时 |
| 只需多轴平滑 S 曲线、jerk 限制轨迹 | Ruckig | 许可为 MIT（非文章所写 MPL-2.0）；带中间路点的实时计算需 Pro |
| 机械臂动力学、力矩控制、DDP 优化 | Pinocchio + Crocoddyl | 一致 |
| 单片机无刷 FOC 电机环 | SimpleFOC | 一致 |
| ROS2 机械臂项目 | MoveIt2 + ros2_control + Ruckig | 一致；ros2_control 为 Apache-2.0 |
| 步进电机单片机 | FastAccelStepper | S 型加减速、无浮点两点未核对 |

## 关联页面

- [全身控制](../concepts/whole-body-control.md)
- [EtherCAT 协议](../concepts/ethercat-protocol.md)
- [模型预测控制](../methods/model-predictive-control.md)

## 参考来源

- [wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md](../../sources/blogs/wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md)
