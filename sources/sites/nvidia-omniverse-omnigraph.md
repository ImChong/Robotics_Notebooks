# OmniGraph — Omniverse Extensions 官方文档

> 来源归档

- **标题：** OmniGraph — Omniverse Extensions
- **类型：** site（官方文档）
- **URL：** <https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html>
- **门户：** <https://docs.omniverse.nvidia.com/extensions/latest/index.html>
- **文档更新：** 2026-09-04（页面 footer）
- **入库日期：** 2026-09-05
- **一句话说明：** Omniverse 的可视化脚本框架：在静态 USD 世界上挂行为与交互；统一承载 Action Graph（事件驱动）与 Push Graph（连续求值）等多类图系统。
- **沉淀到 wiki：** 是 → [`wiki/entities/omnigraph.md`](../../wiki/entities/omnigraph.md)

## 开源边界（步骤 2.5）

OmniGraph 随 **Omniverse Kit / Isaac Sim** 分发，不是独立 PyPI 包。实现位于 [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) 与 Omniverse Kit 扩展生态；Python 入口为 `omni.graph.core`。→ **已开源**（随 Isaac Sim 仓 Apache-2.0）。

## 页面要点（2026-09-05）

### 定位

OmniGraph is the **visual scripting language of Omniverse**. It allows the worlds in Omniverse to come alive with behavior and interactivity. It addresses deformers, particles, event-based graphs, and more — **not a single graph type**, but a **composition of many graph systems** under one framework.

### 两类主图（graph type / evaluation type）

| 类型 | 行为 |
|------|------|
| **Action Graph** | 事件驱动（event-driven）行为 |
| **Push Graph** | 节点连续求值（continuous evaluation） |

另有 Particle System、Skeletal Animation 等 **node libraries**。

### 核心概念（Core Concepts）

| 概念 | 说明 |
|------|------|
| **Authoring Graph** | 用户在编辑器中搭建的图；口语中「OmniGraph」多指此层 |
| **Execution Graph** | 运行时实际执行的图表示 |
| **Node** | 由 node type 定义；含 input / output / state **Attributes** |
| **Attribute** | 具名、具类型的数据槽；可连线形成求值网络 |
| **Connection** | 有向依赖边，连接两节点上的特定 Attribute |

架构宣称可在单机与多节点数据中心间 **同一图表示** 扩展，无需改图结构。

### 文档分区（导航）

- **Getting Started：** Core Concepts、Intro to OmniGraph
- **Interface：** OmniGraph Editor、Node Library
- **Tutorials：** Event / Flow Control / Variant / Maneuver nodes
- **Developer：** OmniGraph developer portal（自定义节点开发）

## 对 wiki 的映射

- 实体页：[`wiki/entities/omnigraph.md`](../../wiki/entities/omnigraph.md)
- 交叉：[`wiki/entities/nvidia-omniverse.md`](../../wiki/entities/nvidia-omniverse.md)、[`wiki/entities/isaac-sim.md`](../../wiki/entities/isaac-sim.md)
- Isaac 侧教程归档：[`sources/sites/nvidia-isaac-sim-omnigraph.md`](./nvidia-isaac-sim-omnigraph.md)
