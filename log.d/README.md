# log.d/ — 日志碎片

PR **不要直接修改根目录 `log.md`**（并行 PR 都在其顶部插入，必然合并冲突）。
改为在本目录新增一个碎片文件，内容就是一条完整的 `log.md` 条目：

```bash
make log OP=ingest DESC="sources/papers/xxx.md — 描述"   # 自动生成唯一文件名
```

手写时文件名用 `YYYY-MM-DD-<任意唯一后缀>.md`（按文件名排序即时间顺序），内容格式见 [schema/log-format.md](../schema/log-format.md)。

合入 main 后，`export.yml` 会运行 `scripts/fold_log_fragments.py` 把碎片并入 `log.md` 顶部并删除碎片。
