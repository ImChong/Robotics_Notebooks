# RFC 8878 — Zstandard Compression and the 'application/zstd' Media Type

> 来源归档

- **标题：** RFC 8878 — Zstandard Compression and the 'application/zstd' Media Type
- **类型：** site（IETF 规范）
- **链接：** https://datatracker.ietf.org/doc/html/rfc8878
- **RFC Editor：** https://www.rfc-editor.org/info/rfc8878
- **状态：** **Informational**（非 Internet Standards Track）；取代 RFC 8478
- **入库日期：** 2026-09-24
- **一句话说明：** Zstandard **无损压缩格式** 的 normative 描述：Frame/Block 结构、压缩算法步骤、MIME **`application/zstd`** 与 **Content-Encoding: zstd**；引用开源参考实现 [facebook/zstd](https://github.com/facebook/zstd)。
- **沉淀到 wiki：** 是 → [`wiki/entities/zstandard.md`](../../wiki/entities/zstandard.md)

## 为什么值得保留

- 选型、互操作、合规审计时，**RFC 8878** 是 zstd 格式的 **一手规范**（相对博客与 README 摘要）。
- 与 [gzip RFC 1952](https://datatracker.ietf.org/doc/html/rfc1952) 对照，理解 HTTP/存储层 **Content-Encoding** 注册。

## 核心摘录

### Abstract

- Zstandard（**zstd**）为 **无损** 压缩机制；本文描述格式并注册 MIME media type、content encoding 与 structured syntax suffix。
- 名称含 “standard” 但 **本文档本身不是 Standards Track**。

### 算法与格式要点（§3）

- 目标：**CPU/OS/FS/字符集无关**；适用于 **文件、管道、流式**；中间存储 **有界**，可长流顺序处理。
- 可选 **xxHash-64** 校验（[XXHASH]）。
- **不保证随机访问** 压缩数据。
- **Frame** 彼此独立可单独解压；多帧拼接 = 解压内容拼接。
- **Block** 依赖前序 block 解码，但可 **不等待后继 block** 即开始解压 → 支持 streaming。
- 参考实现： portable C，[ZSTD] 指向 facebook/zstd 开源仓。

### 术语（§2）

- **uncompressed / compressed / decompressed**；**encode/decode**；**frame** vs **block** 边界与依赖关系。

## 开源核查

| 项 | 状态 |
|----|------|
| 规范文本 | **公开**（IETF） |
| 参考实现 | **已开源** — 见 [repos/zstd.md](../repos/zstd.md) |

## 对 wiki 的映射

- [Zstandard（实体）](../../wiki/entities/zstandard.md)
- [Zstandard 仓库归档](../repos/zstd.md)
