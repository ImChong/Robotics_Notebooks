# codec4AI 官方代码

> 来源归档（ingest）

- **仓库：** <https://github.com/kaname-madoka18/codec4AI>
- **核查版本：** `f9f9c6d6197db7281fe9c7664dbf2b07d588ce6b`；2026-10-30
- **许可：** GPL-3.0-or-later；依赖许可见仓库 NOTICE。
- **论文：** <https://arxiv.org/abs/2610.33079>；[论文归档](../papers/codec4ai_arxiv_2610_33079.md)
- **项目页：** <https://skywalker-yqz.github.io/codec4AI/>；[站点归档](../sites/codec4ai.md)
- **知识节点：** [codec4AI](../../wiki/entities/paper-codec4ai.md)

## 实现与入口

`examples/quickstart.py` 构造合成帧，调用 `video_loader.transcode(..., mode="fast264")`，再对重复且乱序索引调用 `decode`。`src/video_loader/_api.py` 调用 `_native`，`src/native/native.cpp` 的 `decode_native` 建索引、构建参考图和 BFS 闭包，进而执行解码；其余帧走 state-only。无补丁的 FFmpeg 退回连续解码；AV1 当前采用完整包集合。预编译 wheel 的平台和版本限制以 README 与 release notes 为准。公开 fast 模式 GOP 为 8，与论文主实验 GOP 32 配置不可直接等同。
