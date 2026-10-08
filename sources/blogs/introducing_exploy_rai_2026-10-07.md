# Introducing Exploy: Simplifying RL Policy Deployment in Autonomous Robotics

> 来源归档（blog / Robotics and AI Institute 官方）

- **标题：** Introducing Exploy: Simplifying RL Policy Deployment in Autonomous Robotics
- **类型：** blog / 官方技术介绍
- **发布：** 2026-10-07
- **链接：** <https://rai-inst.com/resources/blog/introducing-exploy-simplifying-rl-policy-deployment-in-autonomous-robotics/>
- **机构：** Robotics and AI Institute（RAI Institute）
- **代码：** <https://github.com/rai-opensource/exploy>（MIT，已公开）
- **文档：** <https://rai-opensource.github.io/exploy/>
- **入库日期：** 2026-10-08
- **一句话说明：** RAI 介绍 Exploy 如何把仿真环境中的观测生成、策略前向与动作处理统一导出为 ONNX 计算图，并由 C++/ROS 控制器接入机器人状态与命令接口。
- **沉淀到 wiki：** 是 → [`wiki/entities/exploy.md`](../../wiki/entities/exploy.md)

---

## 核心论点

普通策略导出通常只包含神经网络结构与权重，仿真侧的状态到观测计算、动作缩放/限幅等仍需在 C++ 中重新实现。Exploy 以 PyTorch tracing 跟踪环境和策略的可导出计算，将这些逻辑合入自包含的 ONNX 图，目标是减少两边代码分叉、静默 bug 与部署迭代成本。

## 两个组件

| 组件 | 职责 |
|------|------|
| **Exporter** | 跟踪环境 observation、actor 前向、action processing；登记 inputs、outputs、循环策略 memory 和控制元数据 |
| **Controller** | 基于 ONNX Runtime 的轻量 C++ 控制器；将模型张量 I/O 匹配到机器人状态、命令与数据采集接口 |

博客所说的端到端是计算管线覆盖面：平台特定传感器读取、总线/驱动与执行器安全策略仍需要对接。官方文档还提供导出前后数值对齐评估，因此应读作“提供验证机制”，而不是任何模型都自动与 PyTorch 数值完全相同。

## RAI 报告的部署实例

- Roadrunner：双足轮式平台，在不同移动模式间切换；博客称 Exploy 承载统一控制与起身策略从仿真零样本迁移。
- Spot：深度观测与循环策略用于高障碍和 parkour 地形。
- RAI 的 UMV、自主越野自行车平台，以及 Unitree G1、Boston Dynamics Atlas 也被列为使用场景。

以上是 RAI 官方文章中的部署陈述；文章未给出统一 benchmark、成功率或延迟数据，不应据此推导定量性能提升。

## 对 wiki 的映射

- [Exploy 项目实体](../../wiki/entities/exploy.md) — 统一整理方法、运行入口、边界与应用证据。
- [Exploy 官方源码归档](../repos/rai-opensource-exploy.md)
- [Exploy 官方文档站归档](../sites/exploy-docs.md)