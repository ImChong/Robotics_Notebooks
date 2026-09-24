---
type: entity
tags: [serialization, logging, robotics, dataset, io, hmi-opensource-table, linux-foundation]
status: complete
updated: 2026-09-24
summary: "MCAP：Foxglove 开源、序列化无关的 timestamped pub/sub 日志容器；Schema/Channel/Message + 可选 Chunk 索引；ROS2/Protobuf/JSON 载荷；真机与 Isaac 栈常用，可转 LeRobot。"
related:
  - ../concepts/hdf5-file-format.md
  - ../concepts/lerobot-dataset-v3.md
  - ../comparisons/hdf5-mcap-lerobot-data-formats.md
  - ../entities/foxglove-studio.md
  - ../entities/plotjuggler.md
  - ../entities/isaac-gr00t.md
  - ../entities/isaac-teleop.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/sites/mcap-spec.md
  - ../../sources/repos/mcap-log-format.md
---

# MCAP

## 一句话定义

**MCAP**（[foxglove/mcap](https://github.com/foxglove/mcap)，**MIT**）是 **模块化日志容器文件格式**：在单文件中记录 **带时间戳的 pub/sub 消息**，**载荷序列化方式任意**（ROS 2、Protobuf、JSON 等由 **Schema** 描述）；支持 **Chunk + Summary 索引** 实现高吞吐写入与 **按时间/主题随机读**，广泛用于机器人 **真机录制、复盘与格式转换**（如 **MCAP → LeRobot**）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MCAP | MCAP | 格式名；magic 字节 `\\x89MCAP0\\r\\n` |
| ROS 2 | Robot Operating System 2 | 常见 MCAP 载荷之一（CDR） |
| CLI | Command-Line Interface | `mcap` 检查/合并/切分 |
| API | Application Programming Interface | Python `mcap`、TS `@mcap/core` 等 |
| IL | Imitation Learning | 日志常经转换进入训练集 |
| HF Hub | Hugging Face Hub | LeRobot 数据集托管（对比格式） |

## 为什么重要

- **一条文件多通道：** 相机、关节、IMU、指令可 **分 Channel** 写入同一 MCAP，Foxglove / PlotJuggler 按时间轴对齐 — 比「每模态一个文件」更易运维。
- **序列化无关：** 同容器可混 ROS 2 topic 与自定义 Protobuf；换中间件不必换容器格式（换 Schema/Channel 定义）。
- **真机 vs 训练格式桥：** [Isaac GR00T](../entities/isaac-gr00t.md) 真机参考流 **MCAP → LeRobot → LEAPP**；[Gen-HumanEgo](../entities/gen-human-ego-dataset.md) 等发布 **MCAP episode**。
- **与 HDF5 / LeRobot 三角关系：** MCAP = **录制/日志**；HDF5 = **仿真/IL 中间数组**；LeRobot = **Hub 训练数据集** — 见 [对比页](../comparisons/hdf5-mcap-lerobot-data-formats.md)。

## 核心原理

### 文件结构（mcap.dev/spec）

```mermaid
flowchart LR
  M1[Leading Magic]
  H[Header op=0x01]
  D[Data section<br/>Schema Channel Message Chunk…]
  DE[Data End]
  S[Summary optional]
  SO[Summary Offset optional]
  F[Footer op=0x02]
  M2[Trailing Magic]
  M1 --> H --> D --> DE --> S --> SO --> F --> M2
```

| 段 | 要点 |
|----|------|
| **Magic** | 首尾固定；第 5 字节 `0x30`（ASCII `'0'`）= **major version** |
| **Data** | Message 可直接写或在 **Chunk** 内；**Data End 必须最后** |
| **Summary** | 可选；**Chunk Index** 存在时 Message **应进 Chunk**（否则索引漏读） |
| **Footer** | 索引读者入口；配合 Summary Offset **随机访问** |

### 核心记录（概念）

- **Schema** — 消息编码说明（如 ROS2 msg、Protobuf）  
- **Channel** — 主题/流 ID，绑定 Schema  
- **Message** — log time + publish time + payload  
- **Chunk** — 压缩与索引单元  

私有 opcode **0x80–0xFF** 留给应用扩展。

### 生态（官方 README）

| 语言 | 包 |
|------|-----|
| Python | PyPI **`mcap`** |
| C++ | Conan **`mcap`** |
| TS | **`@mcap/core`** |
| Go / Rust / Swift | 见仓库子目录 |

**CLI：** `brew install mcap` 或 GitHub **releases** — inspect、merge、split。

## 工程实践

### 开源状态（2026-09-24）

- **已开源**：[github.com/foxglove/mcap](https://github.com/foxglove/mcap) **MIT**  
- **规范一手：** [mcap.dev/spec](https://mcap.dev/spec)（归档 [sources/sites/mcap-spec.md](../../sources/sites/mcap-spec.md)）  
- **可视化：** [Foxglove Studio](./foxglove-studio.md)、[PlotJuggler](./plotjuggler.md)（MCAP 读取）

### 机器人管线中的位置

1. **录制：** 真机或中间件桥写 MCAP（高吞吐、可索引）  
2. **质检：** CLI `mcap info` / Studio 回放  
3. **转换：** 项目脚本 **MCAP → LeRobot**（字段映射因栈而异）  
4. **训练：** [LeRobotDataset v3](../concepts/lerobot-dataset-v3.md) + `lerobot-train`

## 局限与风险

- **不是训练数据集格式：** 需转换层；无 LeRobot 式 **`meta/stats.json`** 归一化约定  
- **Schema 多样性：** 读 MCAP 须带对 **Schema 实现**（ROS 2 类型等）  
- **索引约束：** 混用「chunk 内/外 Message」会破坏索引读者假设  
- **与 rosbag2：** ROS 2 默认 bag 格式可能不同；转换工具链需单独验证

## 关联页面

- [HDF5 文件格式](../concepts/hdf5-file-format.md)
- [LeRobotDataset v3.0](../concepts/lerobot-dataset-v3.md)
- [HDF5 vs MCAP vs LeRobot](../comparisons/hdf5-mcap-lerobot-data-formats.md)
- [Isaac Teleop](./isaac-teleop.md)
- [Isaac GR00T](./isaac-gr00t.md)

## 参考来源

- [sources/sites/mcap-spec.md](../../sources/sites/mcap-spec.md)
- [sources/repos/mcap-log-format.md](../../sources/repos/mcap-log-format.md)

## 推荐继续阅读

- [MCAP Format Specification](https://mcap.dev/spec)
- [MCAP CLI 指南](https://mcap.dev/guides/cli)
- [MCAP README（GitHub）](https://github.com/foxglove/mcap/blob/main/README.md)
