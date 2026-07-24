---
type: entity
tags: [repo, unitree, unitreerobotics, util]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-actuator-sdk.md
  - ./unitree-sdk2.md
sources:
  - ../../sources/repos/logging-mp.md
  - ../../sources/repos/unitree.md
summary: "面向多进程的轻量 Python 日志工具（rich 终端 + 文件写入）。"
---

# logging-mp

**logging-mp** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **工具与调试** 主线。

## 一句话定义

面向多进程的轻量 Python 日志工具（rich 终端 + 文件写入）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| API | Application Programming Interface | 应用程序编程接口 |
| SDK | Software Development Kit | 软件开发工具包 |
| RL | Reinforcement Learning | 强化学习 |
| DDS | Data Distribution Service | 分布式实时通信中间件 |
| ONNX | Open Neural Network Exchange | 跨框架神经网络交换格式 |

## 为什么重要

调试与日志工具虽小，却是多进程训练/部署排障的高频依赖。

在宇树官方开源地图中，本仓是 **工具与调试** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/logging-mp`](https://github.com/unitreerobotics/logging-mp) |
| 组织分类 | 工具与调试 |
| 星标（2026-07-24） | ~3 |
| 最近推送 | 2026-03-24 |
| 主要语言 | Python |

## 工程实践

- ⚡ Support Multi-Processing & Thread: Fully compatible with threading modules. Child processes automatically send logs to the main process.
- 💻 Cross-Platform Support: Works seamlessly with both fork (Linux) and spawn (Windows/macOS) start methods.
- 🎨 Rich Integration: Beautiful, colorized console output powered by Rich.
- 📂 File Logging: Aggregates logs from all processes and threads into timestamped log files with size-based rollover and count-based cleanup.

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/logging-mp.md](../../sources/repos/logging-mp.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [unitree_actuator_sdk](./unitree-actuator-sdk.md)
- [unitree_sdk2](./unitree-sdk2.md)

## 参考来源

- [sources/repos/logging-mp.md](../../sources/repos/logging-mp.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/logging-mp>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
