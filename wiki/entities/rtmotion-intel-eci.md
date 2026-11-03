---
type: entity
project_id: rtmotion-intel-eci
tags:
  - plcopen
  - motion-control
  - cpp
  - ethercat
  - intel-eci
status: complete
updated: 2026-11-03
related:
  - ../entities/igh-ethercat-master.md
  - ../entities/plcopen-pyplcopen.md
  - ../entities/ruckig.md
  - ../concepts/ethercat-protocol.md
  - ../overview/cpp-motion-control-open-source-libraries-map.md
sources:
  - ../../sources/repos/rtmotion_intel_eci.md
  - ../../sources/blogs/wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md
summary: "RTmotion 是 Intel 创建的开源 C++ 库，实现 PLCopen 运动控制标准第 1、2 部分；在 Intel ECI 中与 IgH EtherCAT Master 和 SoftPLC 配合构建通用运动控制应用。"
---

# RTmotion（Intel ECI）

**RTmotion（Intel ECI）**（（未能定位公开仓库））是 [Qt历险记《C/C++ 运动控制开源库分类整理》](https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw) 中「工业总线与控制栈」一类的项目。本页以**一手资料**为准，文章说法放在「文章说法核对」里对照。

## 一句话定义

**RTmotion 是 Intel 创建的开源 C++ 库，实现 PLCopen 运动控制标准第 1、2 部分；在 Intel ECI 中与 IgH EtherCAT Master 和 SoftPLC 配合构建通用运动控制应用。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OTG | Online Trajectory Generation | 在线轨迹生成 |
| PTP | Point-To-Point | 点到点运动 |

## 为什么重要

- 在文章的选型建议里属于「工业总线与控制栈」；横向对比见 [C/C++ 运动控制开源库地图：轨迹生成、动力学、工业总线与 ROS 栈（一手资料核对版）](../overview/cpp-motion-control-open-source-libraries-map.md)。

## 一手资料

| 项 | 内容 |
|----|------|
| **仓库 / 官方来源** | （未能定位公开仓库） |
| **许可（核查）** | 文章写「开源」；本次未能定位公开仓库，许可未核实 |
| **最近提交** | 未核实 |
| **论文 / 文档** | Intel Edge Controls for Industrial（ECI）官方文档：PLCopen Motion Control（eci.intel.com/docs/3.3/components/plcopen-motion.html）与 RTmotion 文档（eci.intel.com/rt-motion-docs/index.html） |

**一手描述：** ECI 官方文档（3.3）：「RTmotion is an open source C++ library created by Intel」，实现 PLCopen motion control standard 的 part 1 和 2，加上机器视觉与 ROS 2 扩展后已被部署用于离散自动化设备、AMR 与机械臂；通过 ECI 的 apt 仓库分发（含 `plcopen-motion`、`plcopen-servo`、`plcopen-ruckig`、`plcopen-databus` 等包）。

## 文章说法核对

- **文章：** 语言 C++；许可「开源」；PLCopen 运动功能块，和 IgH EtherCAT Master 集成，适合工业自动化、AMR、机械手。
- **未能定位公开 GitHub 仓库**：我尝试了常见仓库路径与搜索，没有找到可访问的 RTmotion 源码仓库；一手资料只有 Intel ECI 官方文档，因此**许可与最近提交均未核实**，本页不编造仓库地址。
- 文档显示 ECI 另有 `plcopen-ruckig-dev` 包，即 RTmotion 体系可与 Ruckig 配合；该关系来自官方文档的包列表。

## 关联页面

- [igh-ethercat-master](../entities/igh-ethercat-master.md)
- [plcopen-pyplcopen](../entities/plcopen-pyplcopen.md)
- [ruckig](../entities/ruckig.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [cpp-motion-control-open-source-libraries-map](../overview/cpp-motion-control-open-source-libraries-map.md)

## 参考来源

- [rtmotion_intel_eci.md](../../sources/repos/rtmotion_intel_eci.md)
- [wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md](../../sources/blogs/wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw)
