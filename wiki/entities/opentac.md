---
type: entity
project_id: opentac
tags: [tactile-sensing, tactile-sensor, manipulation, open-source, simulation]
status: active
updated: 2026-10-23
related:
  - ../concepts/tactile-sensing.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/visuo-tactile-fusion.md
  - ./eflesh.md
  - ./paper-vtap-gripper.md
sources:
  - ../../sources/sites/opentac-project.md
  - ../../sources/repos/opentac.md
summary: "OpenTac 是一套开放的定制化触觉皮肤生态：从柔性 PCB 与可打印剪切凸点设计，到读取、仿真、跨皮肤数据迁移和自适应采样；2×2 taxel 凸点以共模/差分响应估计法向力与剪切力。"
---

# OpenTac（可定制 3D 触觉皮肤生态）

**OpenTac** 是面向机器人任意可覆盖表面的开放触觉感知工具链。它将皮肤设计、制造、传感器读取、仿真、跨传感器数据映射和自适应采样放在一个共享 manifest 驱动的生态中；项目提供 [在线 CAD 设计器](https://opentac.github.io/cad/) 与 [官方代码](https://github.com/opentac/opentac)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CAD | Computer-Aided Design | 从轮廓或曲面生成传感器版图与制造文件 |
| PCB | Printed Circuit Board | OpenTac 使用柔性 PCB 承载交错电极阵列 |
| TPU | Thermoplastic Polyurethane | 可打印的弹性材料，常用于剪切感知凸点外壳 |
| DRC | Design Rule Check | 检查版图走线间距、边界等制造规则 |

## 核心结构与力估计

皮肤由柔性 PCB 上的交错电极 taxel 阵列、覆盖其上的压阻薄膜，以及外层结构组成。用于三维力估计时，外壳在每个 **2×2 taxel** 区域上方放置一个柔性凸点：

- **法向力：** 向下按压使四个 taxel 产生近似均匀的压力变化，四路响应的共模部分用于估计法向力。
- **剪切力：** 横向载荷使凸点倾斜，凸点偏向一侧的两个 taxel 受力增加、另一侧减小；两组之间的差分响应给出剪切方向和分量。
- **标定：** 官方说明可用参考力传感器将四路读数标定为牛顿值；未标定的原始读数不应直接当作力值。

## 全栈组成

| 模块 | 用途 |
|------|------|
| OpenTac-CAD | 从 2D 轮廓、DXF 或可展开的 STL 曲面生成 taxel 布局、走线、连接器、Gerber 和打印外壳文件 |
| Readout boards / firmware | 通过 USB 读取柔性皮肤；官方提供最高 30×30 与 15×15 taxel 规模的读出板方案 |
| OpenTac-Sim | 基于传感器 manifest 仿真凸点与触觉响应，便于接入机器人仿真 |
| OpenTac-Transfer | 按传感器物理坐标将触觉数据从一种皮肤布局映射到另一种布局 |
| OpenTac-Xampling | 根据接触稀疏性自适应选择读取 taxel，提升局部接触事件的时间分辨率 |

## 采样速率应如何理解

官方基准在 1024-taxel 皮肤上测得：完整 raster 扫描为 **67 frames/s**；Xampling 在每次更新读取 16 个 taxel 时为 **1286 partial frames/s**。因此“千赫兹级”指稀疏接触下的部分阵列更新率，不等于每次更新完整扫描所有 taxel；采样预算增加或接触面积扩大时，更新率会下降。

这类自适应读取适合捕获毫秒级短接触。若下游策略需要完整触觉帧，应同时考虑全阵列扫描率、未采样区域状态和采样掩码，不能把未读取 taxel 的旧值当成当前帧的新测量。

## 制造与复现边界

- 官方项目页公开项目代码仓库，声明代码采用 **MIT** 许可证、硬件采用 **CERN-OHL-P-2.0**、文档和媒体采用 **CC BY 4.0**。
- CAD 可在线使用，也可下载代码本地运行。制造仍需采购或制作柔性 PCB、压阻薄膜、读出板，并打印弹性剪切凸点；项目不是无需装配的商用传感器模块。
- 剪切估计依赖凸点与对应 2×2 taxel 的几何对齐以及材料/结构标定。更换薄膜、凸点几何或读出设置后，应重新验证力映射。

## 与其他触觉路线的关系

- [触觉感知](../concepts/tactile-sensing.md) 总结触觉模态与传感器路线；OpenTac 是柔性压阻阵列结合机械凸点的 normal/shear sensing 实例。
- [eFlesh](./eflesh.md) 使用磁性微结构估计多轴触觉量，感知机理与 OpenTac 的压阻差分阵列不同。
- [VTAP Gripper](./paper-vtap-gripper.md) 展示了压阻触觉阵列在夹爪上的应用；OpenTac 的重点则是面向多种表面的可定制设计和软硬件工具链。

## 推荐继续阅读

- [OpenTac 官方项目页](https://opentac.github.io/)
- [OpenTac-CAD 设计器](https://opentac.github.io/cad/)
- [OpenTac 论文页](https://opentac.github.io/paper/)
- [OpenTac 官方代码仓库](https://github.com/opentac/opentac)

## 参考来源

- [OpenTac 官方项目页与论文摘要](../../sources/sites/opentac-project.md)
- [OpenTac 官方代码仓库归档](../../sources/repos/opentac.md)
