---
type: entity
project_id: plcopen-pyplcopen
tags:
  - plcopen
  - motion-control
  - cpp
  - ethercat
  - ot
status: complete
updated: 2026-11-03
code: https://github.com/lusipad/plcopen
related:
  - ../concepts/ethercat-protocol.md
  - ../entities/soem.md
  - ../entities/igh-ethercat-master.md
  - ../entities/ruckig.md
  - ../overview/cpp-motion-control-open-source-libraries-map.md
sources:
  - ../../sources/repos/plcopen_pyplcopen.md
  - ../../sources/blogs/wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md
summary: "plcopen（PyPI 名 pyplcopen）是现代 C++17 header-only 的 PLCopen 风格运动控制内核，含时间最优 OTG、前瞻规划、轴与组状态机、功能块，以及 IEC 61131-3 ST 编译器 + 确定性字节码 VM；Python 包是其仿真门面。"
---

# pyplcopen / plcopen 运动控制内核

**pyplcopen / plcopen 运动控制内核**（[https://github.com/lusipad/plcopen](https://github.com/lusipad/plcopen)）是 [Qt历险记《C/C++ 运动控制开源库分类整理》](https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw) 中「轨迹规划库」一类的项目。本页以**一手资料**为准，文章说法放在「文章说法核对」里对照。

## 一句话定义

**plcopen（PyPI 名 pyplcopen）是现代 C++17 header-only 的 PLCopen 风格运动控制内核，含时间最优 OTG、前瞻规划、轴与组状态机、功能块，以及 IEC 61131-3 ST 编译器 + 确定性字节码 VM；Python 包是其仿真门面。**

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
| **仓库 / 官方来源** | [https://github.com/lusipad/plcopen](https://github.com/lusipad/plcopen) |
| **许可（核查）** | Apache-2.0（LICENSE：Apache License 2.0；README「许可：Apache License 2.0」） |
| **最近提交** | 2026-11-02 |
| **论文 / 文档** | 无论文；PyPI 项目页 pypi.org/project/pyplcopen（摘要：PLCopen motion control kernel — Python simulation facade） |

**一手描述：** README：C++17 header-only 运动控制内核；实时基础设施、时间最优 OTG 与前瞻规划、轴 / 组状态机、PLCopen 风格功能块与 ST 编译器 + 字节码 VM；周期路径零分配 / 零锁 / 无异常；是可嵌入库而非完整 PLC 运行时；PyPI 包为 `pyplcopen`，作者 lusipad。

## 文章说法核对

- **文章：** 语言 C++17 header-only、Python 封装；许可 MIT；PLCopen 运动内核（PTP、直线、圆弧、前瞻、连续插补、电子凸轮、同步、回零、轴组状态机），零分配实时周期；适用自研运动控制器、EtherCAT 上位机，纯算法不含总线驱动。
- **名称对应**：文章写的「pyplcopen」是 PyPI 包名，真实源码仓库是 **lusipad/plcopen**（PyPI 页面 Repository 链接指向它）；Python 侧是仿真门面，内核是 C++17。
- **许可不一致**：文章写 MIT；仓库 LICENSE 与 README 均为 **Apache-2.0**。
- **EtherCAT 尚未完成**：README 状态表写 EtherCAT「🔴 未开工」、是最大剩余软件块；文章把它列为「EtherCAT 上位机」适用场景，目前只能与外部 EtherCAT 主站搭配。
- README 写最新正式版为 v0.21.0（PyPI 页面仍显示 0.21.0）；项目较新、版本 0.x，文章未提成熟度。

## 关联页面

- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [soem](../entities/soem.md)
- [igh-ethercat-master](../entities/igh-ethercat-master.md)
- [ruckig](../entities/ruckig.md)
- [cpp-motion-control-open-source-libraries-map](../overview/cpp-motion-control-open-source-libraries-map.md)

## 参考来源

- [plcopen_pyplcopen.md](../../sources/repos/plcopen_pyplcopen.md)
- [wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md](../../sources/blogs/wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw)
