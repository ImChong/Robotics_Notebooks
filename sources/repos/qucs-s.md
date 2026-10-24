---
type: source
source_type: repo
title: "Qucs-S（官方源码）"
url: "https://github.com/ra3xdh/qucs_s"
project_id: qucs-s
project: "https://ra3xdh.github.io/"
code: "https://github.com/ra3xdh/qucs_s"
license: GPL-2.0
status: public
updated: 2026-10-24
summary: "Qucs-S 官方源码：基于 Qt 的原理图捕获与电路仿真 GUI，支持连接多种外部仿真内核。"
---

# Qucs-S（官方源码仓库）

> 来源归档

- **类型：** repo
- **主仓库：** https://github.com/ra3xdh/qucs_s
- **项目主页：** https://ra3xdh.github.io/
- **文档：** https://qucs-s-help.readthedocs.io/en/latest/
- **许可证：** GPL-2.0（仓库 COPYING）
- **开源状态：** 已开源；本仓库包含 GUI 源码。Ngspice、Xyce、SpiceOpus 等仿真后端是外部依赖，需按平台安装或配置。
- **一句话说明：** Qucs-S 提供原理图编辑、仿真控制与结果可视化，将电路网表交给所选仿真后端运行。
- **沉淀到 wiki：** [Qucs-S](../../wiki/entities/qucs-s.md)

---

## 仓库定位

官方 README 将 Qucs-S 定位为基于 Qt 的电路仿真 GUI，支持的内核包括 Ngspice（推荐）、Xyce、SpiceOpus，以及非 SPICE 内核 Qucsator。仓库提供 CMake 构建说明；当前开发分支依赖 Qt 6。

Qucs-S 并非独立求解器：安装 GUI 后仍需配置相应仿真后端。后端能力、模型兼容性和分析类型取决于具体内核。

## 一手来源

- [官方 README](https://github.com/ra3xdh/qucs_s/blob/current/README.md)
- [官方源码仓库](https://github.com/ra3xdh/qucs_s)
- [许可证文件 COPYING](https://github.com/ra3xdh/qucs_s/blob/current/COPYING)
- [官方 Wiki：项目简介与主要功能](https://github.com/ra3xdh/qucs_s/wiki)
- [Qucs-S 官方文档](https://qucs-s-help.readthedocs.io/en/latest/)
- [项目主页归档](../sites/qucs-s-official.md)

## 交叉链接

- Wiki 实体：[Qucs-S](../../wiki/entities/qucs-s.md)
- 官方项目主页归档：[qucs-s-official.md](../sites/qucs-s-official.md)
