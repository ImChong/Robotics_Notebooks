# LZ4（lz4/lz4）

> 来源归档

- **标题：** LZ4 — Extremely fast compression
- **类型：** repo
- **作者：** Yann Collet（@Cyan4973）及社区
- **链接：** https://github.com/lz4/lz4
- **主页：** http://www.lz4.org / https://lz4.github.io/lz4/
- **许可：** BSD 2-Clause
- **入库日期：** 2026-09-24
- **一句话说明：** 面向 **极高吞吐** 的无损压缩：单核压缩常 **>500 MB/s**、解压可达 **数 GB/s**；提供默认 LZ4、高压缩 **LZ4_HC**、**LZ4 Frame** 流式封装，并与 Zstd 字典训练 interoperable。
- **沉淀到 wiki：** 是 → [`wiki/entities/lz4.md`](../../wiki/entities/lz4.md)

## 开源核查（2026-09-24）

| 项 | 状态 |
|----|------|
| 参考实现 C 库 + `lz4` CLI | **已开源** — BSD 2-Clause |
| 格式文档 | **公开** — `doc/lz4_Block_format.md`、`doc/lz4_Frame_format.md` |

## 核心摘录（README，dev 分支）

### 定位

- **无损**；压缩速度 **>500 MB/s/核**，多核可扩展。
- **解压极快**（多 GB/s/核），多核上常触 RAM 带宽上限。
- **`acceleration`** 可在压缩比与速度间动态权衡。
- **LZ4_HC** 用更多 CPU 换更高压缩比；**各变体解压速度相同**。
- 支持 **字典压缩**（API + CLI）；字典末 **64KB** 生效；可与 [Zstd Dictionary Builder](https://github.com/facebook/zstd/blob/dev/programs/zstd.1.md#dictionary-builder) 联用改善小文件比。

### 基准（README 表，Silesia Corpus，Core i7-9700K，lzbench）

| Compressor | Ratio | Compression | Decompression |
|------------|-------|-------------|---------------|
| **LZ4 default (v1.9.0)** | 2.101 | **780 MB/s** | **4970 MB/s** |
| Snappy 1.1.4 | 2.091 | 565 MB/s | 1950 MB/s |
| Zstandard 1.4.0 -1 | 2.883 | 515 MB/s | 1380 MB/s |
| **LZ4 HC -9** | 2.721 | 41 MB/s | 4900 MB/s |

（README 较新版本 zstd/lz4 互引表见 [zstd README](../repos/zstd.md)。）

### 格式分层

- **Block**：原始 LZ4 块格式（`doc/lz4_Block_format.md`）。
- **Frame**：任意长流/文件由多块组成帧（`doc/lz4_Frame_format.md`）；互操作实现须遵守 Frame。

## 对 wiki 的映射

- [LZ4（实体）](../../wiki/entities/lz4.md)
- [Zstandard（实体）](../../wiki/entities/zstandard.md)
- [LZ4 vs Zstandard（选型对比）](../../wiki/comparisons/lz4-vs-zstandard.md)
- [机器人关键帧与运动编辑工具](../../wiki/entities/robot-motion-keyframe-editors.md) — joblib + LZ4 运动包
