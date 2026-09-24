# Zstandard / zstd（facebook/zstd）

> 来源归档

- **标题：** Zstandard — Real-time compression at zlib-level and better ratios
- **短名：** zstd
- **类型：** repo
- **维护：** Meta（原 Facebook）及社区；参考实现作者 Yann Collet
- **链接：** https://github.com/facebook/zstd
- **主页：** https://facebook.github.io/zstd/
- **格式规范：** [RFC 8878](https://datatracker.ietf.org/doc/html/rfc8878)（IETF Informational；归档：[sites/rfc-8878-zstandard.md](../sites/rfc-8878-zstandard.md)）
- **许可：** **BSD OR GPLv2**（双许可 C 库 + CLI）
- **入库日期：** 2026-09-24
- **一句话说明：** 面向 **实时压缩** 的无损算法：默认档位常达 **zlib 级或更好压缩比**，压缩/解压 **数百 MB/s～数 GB/s**；格式 **稳定且 RFC 文档化**；CLI 可读写 `.zst`/`.gz`/`.xz`/`.lz4`；内置 **字典训练** 优化小记录/小文件。
- **沉淀到 wiki：** 是 → [`wiki/entities/zstandard.md`](../../wiki/entities/zstandard.md)

## 开源核查（2026-09-24）

| 项 | 状态 |
|----|------|
| `libzstd` + `zstd` CLI | **已开源** — BSD OR GPLv2 |
| 格式 | **RFC 8878** 公开；多独立实现 |
| 生产部署 | README：**Meta 及多家大型云** 大规模使用；持续 **oss-fuzz** |

## 核心摘录（README，dev 分支）

### 定位

- 熵编码阶段依赖 [Huff0 / FSE](https://github.com/Cyan4973/FiniteStateEntropy)。
- **负压缩级别 `--fast=#`**：更快、更低压缩比。
- **高档位**：更强压缩比、更慢压缩；**解压速度在各档位大致稳定**（与 zlib/lzma 等 LZ 族类似）。

### 基准（Silesia，Core i7-9700K / 文档中另有一组 i7-6700K 曲线）

| Compressor | Ratio | Compression | Decompress |
|------------|-------|-------------|------------|
| **zstd 1.5.7 -1** | 2.896 | 510 MB/s | 1550 MB/s |
| zlib 1.3.1 -1 | 2.743 | 105 MB/s | 390 MB/s |
| **zstd 1.5.7 --fast=4** | 2.146 | 665 MB/s | 2050 MB/s |
| **lz4 1.10.0** | 2.101 | 675 MB/s | 3850 MB/s |

### Small Data / 字典压缩

- 小 payload 难压缩因缺乏「历史」；**`zstd --train`** 生成 **dictionary**，压缩/解压时 **`-D dictionaryName`**。
- 字典收益主要在 **前几 KB**；需 **同分布小样本**（无通用字典）。
- 可与 LZ4 字典 API 联用（见 [lz4 README](../repos/lz4.md)）。

### 构建入口

- 参考构建：**`make`** → `zstd` CLI + `lib/` 下 `libzstd`；亦支持 cmake / meson / vcpkg 等。

## 对 wiki 的映射

- [Zstandard（实体）](../../wiki/entities/zstandard.md)
- [LZ4（实体）](../../wiki/entities/lz4.md)
- [LZ4 vs Zstandard（选型对比）](../../wiki/comparisons/lz4-vs-zstandard.md)
- [RFC 8878 归档](../sites/rfc-8878-zstandard.md)
