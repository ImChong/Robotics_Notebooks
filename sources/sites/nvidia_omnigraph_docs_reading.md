# NVIDIA Omniverse / Isaac Sim OmniGraph 官方文档通读

> 来源归档（site / 官方文档通读）

- **标题：** OmniGraph — Omniverse Extensions & NVIDIA Isaac Sim Documentation
- **机构：** NVIDIA Corporation
- **原文：**
  - <https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html>（Omniverse Extensions OmniGraph 索引）
  - <https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph/getting-started/core_concepts.html>（Core Concepts）
  - <https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph/tutorials/gentle_intro.html>（Introduction to OmniGraph）
  - <https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph/interface.html>（OmniGraph Interface）
  - <https://docs.isaacsim.omniverse.nvidia.com/latest/omnigraph/index.html>（Isaac Sim OmniGraph Tool Reference）
- **入库日期：** 2026-11-28
- **核对范围：** 完整通读上述 5 篇 NVIDIA 官方在线文档页面（文档快照注明 OmniGraph 108.1，页面更新日期 2026-10-14 / 2026-11-24）。`omni.graph.core` 底层 C++ 求值引擎随 Omniverse Kit 二进制闭源分发，不在公开源码仓内。
- **通读：** 已完成
- **一句话说明：** NVIDIA 官方对 Omniverse 统一图计算/可视化编程框架 OmniGraph（Authoring Graph 与 Execution Graph 分离、Action/Push 两种核心求值模式及在 Isaac Sim 内承载 ROS 2 Bridge/Replicator/传感器与控制器的角色）的一手规范说明。

## 一手文档核对要点（2026-11-28）

1. **核心概念与执行模型（`ext_omnigraph/core_concepts.html`）：**
   - **Authoring Graph vs. Execution Graph**：用户在编辑器或 Python 中操作的是描述拓扑的 **Authoring Graph**；运行时由计算后端编译并映射为 **Execution Graph**，支持单机与多节点数据中心扩展而不改动图描述本身。
   - **Node / Attribute / Connection**：节点由 Node Type 定义并包含 `input`、`output` 与 `state` 三类 Attribute；Connection 为连接两个端口的属性有向依赖边。
2. **Action Graph 与 Push Graph 求值差异（`ext_omnigraph.html` & `gentle_intro.html`）：**
   - **Action Graph**（依赖 `omni.graph.bundle.action` 扩展）：事件驱动，沿 Execution 边（如 `On Playback Tick` 的 `outputs:tick` → 写节点的 `Exec In`）按需拉取并求值上游数据依赖（先算上游 `Read Prim Local Transform` / `Get Translation` / `Add` / `Set Translation`，再计算下游 `Write Prim Local Transform`）。
   - **Push Graph**：连续流式推送求值，适合粒子、变形器或每帧必须刷新的连续数据流；另支撑 Particle System 与 Skeletal Animation 图系统。
   - **关键运行时陷阱（`gentle_intro.html`）**：图运行读取与修改的是 **Runtime Data（运行时状态）**，未必即时回写回原始 USD 文件；USD 文件中 authored 的数值仅在图初始化时加载，且更改扩展数据类型端口时必须断开旧连线以重置类型推导。
3. **在 NVIDIA Isaac Sim 中的主干定位（`isaacsim/omnigraph/index.html`）：**
   - 官方定位 OmniGraph 为 Isaac Sim 内 **Replicator（合成数据生成）、ROS 2 Bridge、传感器读取、控制器（如 Differential/Articulation Controller）、外部 I/O 与 UI** 的主引擎。
   - 编辑器入口为 `Window > Graph Editors > Action Graph`（部分 Kit 应用为 `Window > Visual Scripting > Action Graph`），支持 `omni.graph.core` Python 编排（含 On-Demand 手动触发）、自定义 Python/C++/IPC 节点（含 ROS 2 自定义 Python/C++ Node）。

## 对 wiki 与路线的映射

- [`wiki/entities/omnigraph.md`](../../wiki/entities/omnigraph.md) — 实体核心一手依据
- [`wiki/entities/isaac-sim.md`](../../wiki/entities/isaac-sim.md) — Isaac Sim 控制与传感桥接层
- [`roadmap/depth-robot-modeling-digital-twin.md`](../../roadmap/depth-robot-modeling-digital-twin.md) — Stage 4 仿真平台控制图与 Stage 6 SIL 闭环编排
