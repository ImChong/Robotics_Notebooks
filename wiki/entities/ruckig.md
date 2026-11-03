---
type: entity
project_id: ruckig
tags:
  - trajectory-generation
  - motion-control
  - cpp
  - real-time
status: complete
updated: 2026-11-03
code: https://github.com/pantor/ruckig
related:
  - ../methods/trajectory-optimization.md
  - ../entities/ros2-control.md
  - ../entities/moveit2.md
  - ../overview/cpp-motion-control-open-source-libraries-map.md
sources:
  - ../../sources/repos/ruckig.md
  - ../../sources/blogs/wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md
summary: "Ruckig 实时生成从任意初始状态到目标状态（位置、速度、加速度）的、受速度 / 加速度 / jerk 约束的时间最优轨迹，可在线重规划，C++ 为主并带 Python 绑定。"
---

# Ruckig

**Ruckig**（[https://github.com/pantor/ruckig](https://github.com/pantor/ruckig)）是 [Qt历险记《C/C++ 运动控制开源库分类整理》](https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw) 中「轨迹规划库」一类的项目。本页以**一手资料**为准，文章说法放在「文章说法核对」里对照。

## 一句话定义

**Ruckig 实时生成从任意初始状态到目标状态（位置、速度、加速度）的、受速度 / 加速度 / jerk 约束的时间最优轨迹，可在线重规划，C++ 为主并带 Python 绑定。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OTG | Online Trajectory Generation | 在线轨迹生成 |
| PTP | Point-To-Point | 点到点运动 |

## 为什么重要

- 在文章的选型建议里属于「轨迹规划库」；横向对比见 [C/C++ 运动控制开源库地图：轨迹生成、动力学、工业总线与 ROS 栈（一手资料核对版）](../overview/cpp-motion-control-open-source-libraries-map.md)。

## 一手资料

| 项 | 内容 |
|----|------|
| **仓库 / 官方来源** | [https://github.com/pantor/ruckig](https://github.com/pantor/ruckig) |
| **许可（核查）** | MIT（LICENSE：MIT，Lars Berscheid 2021；README 徽章 MIT） |
| **最近提交** | 2026-09-29 |
| **论文 / 文档** | Berscheid & Kröger，Jerk-limited Real-time Trajectory Generation with Arbitrary Target States（RSS XVII 2021，arXiv:2105.04830，README 与 arXiv 均已核对） |

**一手描述：** README：Ruckig 对「state-to-state」运动保证时间最优；允许给中间位置做路点跟随，此时联合计算路径及其时间参数化。

## 文章说法核对

- **文章：** 语言 C++（附 Python 绑定）；许可「社区版 MPL-2.0（商用免费），Pro 版闭源」；带 jerk 限制的时间最优轨迹，在线实时重规划、多轴 PTP、支持中间路径点；替代 Reflexxes；微秒级计算，硬实时友好。
- **许可不一致**：文章写社区版 MPL-2.0；仓库 LICENSE 与 README 徽章均为 **MIT**。
- **中间路径点不是实时的**：README 称社区版带中间路点时会切换到**云 API（非实时）**；要在自己硬件上实时计算需联系作者购买 Ruckig Pro。文章「支持中间路径点 + 硬实时友好」对社区版并不成立。
- README 提到 Pro 版有 `interrupt_calculation_duration`（微秒）等实时保证接口；「微秒级计算」的具体数字本次未复现。

## 关联页面

- [trajectory-optimization](../methods/trajectory-optimization.md)
- [ros2-control](../entities/ros2-control.md)
- [moveit2](../entities/moveit2.md)
- [cpp-motion-control-open-source-libraries-map](../overview/cpp-motion-control-open-source-libraries-map.md)

## 参考来源

- [ruckig.md](../../sources/repos/ruckig.md)
- [wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md](../../sources/blogs/wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw)
