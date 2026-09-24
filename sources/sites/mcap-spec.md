# MCAP Format Specification（mcap.dev）

- **标题：** MCAP Format Specification
- **类型：** site（格式规范）
- **来源：** Foxglove / MCAP 项目
- **链接：** https://mcap.dev/spec
- **仓库：** https://github.com/foxglove/mcap（**MIT**）
- **Kaitai 定义：** https://github.com/foxglove/mcap/blob/main/website/docs/spec/mcap.ksy
- **入库日期：** 2026-09-24
- **一句话说明：** **序列化无关** 的 timestamped pub/sub **日志容器**：Magic + Header/Data/Summary/Footer；**Schema + Channel + Message** 多通道；可选 **Chunk** 索引与压缩；适合机器人 **高吞吐录制与按时间/主题随机读**。
- **沉淀到 wiki：** 是 → [`wiki/entities/mcap-log-format.md`](../../wiki/entities/mcap-log-format.md)

## 开源核查（2026-09-24）

| 项 | 状态 |
|----|------|
| 规范 | **公开**（mcap.dev/spec） |
| 实现 | **已开源** — C++/Go/Python/TS/Rust/Swift 等（见 README 语言表） |
| CLI | **mcap** — `brew install mcap` 或 [releases](https://github.com/foxglove/mcap/releases?q=cli) |

## 核心摘录（官方 Overview & File Structure）

### 定位

> MCAP is a **modular container file format** for recording **timestamped pub/sub messages** with **arbitrary serialization formats**.

### 文件结构

- 首尾 **Magic**：`0x89, M, C, A, P, 0x30, \r, \n`（`0x30` = ASCII `'0'` = **major version**）  
- **Header**（首记录，op=0x01）→ **Data section** → 可选 **Summary** + **Summary Offset** → **Footer**（末记录，op=0x02）→ trailing magic  

### Data section 记录类型

Schema、Channel、Message、Attachment、**Chunk**、Message Index、Metadata、**Data End**（**必须**为 data section 最后一条）

**Chunk 规则（官方）：** 若 Summary 含 **Chunk Index**，Message 应写入 Chunk，否则索引读者可能 **漏掉 chunk 外 Message**。

### Summary section

可选；Grouped by opcode；含 Schema/Channel 副本、Chunk Index、Statistics 等 — 支持 **随机访问**（读者常从 Footer 起读）。

### 记录通用格式

单字节 **opcode** + **uint64 content length** + payload；0x01–0x7F 保留给 MCAP；**0x80–0xFF 私有**；可在记录末 **扩展字段**（Message/DataEnd/Footer 除外）。

### README 补充

- **Serialization-agnostic** — ROS 2 CDR、Protobuf、JSON 等由 Schema 描述  
- **Motivation PDF**：https://mcap.dev/files/evaluation.pdf  
- **Support matrix**：https://mcap.dev/reference  

## 对 wiki 的映射

- [MCAP（实体）](../../wiki/entities/mcap-log-format.md)
- [HDF5 vs MCAP vs LeRobot 对比](../../wiki/comparisons/hdf5-mcap-lerobot-data-formats.md)
- [Foxglove Studio](../../wiki/entities/foxglove-studio.md)
- [PlotJuggler](../../wiki/entities/plotjuggler.md)
