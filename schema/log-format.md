# log.md 格式规范

> 本文件规定 `log.md` 的条目格式与可解析前缀约定。
> log.md 是 Robotics_Notebooks 的 append-only 操作日志，对应 Karpathy LLM Wiki 模式中的运营层记录。

其它 `schema` 文件索引见 [README.md](README.md)。

---

## 基本格式

每条日志以 `##` 标题行开头，格式：

```
## [YYYY-MM-DD] <op> | <描述>
```

- `[YYYY-MM-DD]` — ISO 8601 日期，便于 grep 和排序
- `<op>` — 操作类型（见下方枚举）
- `<描述>` — 一句话说明本次操作内容，包含：影响的文件、关键数字、目的

可在标题行下追加多行正文（`-` 列表），用于记录细节。

---

## Op 类型枚举

| Op | 含义 | 典型场景 |
|----|------|---------|
| `ingest` | 新资料进入知识库 | 新增 sources/papers/*.md、更新 wiki 页参考来源 |
| `query` | 查询并将结论写回 wiki | 新建 wiki/queries/*.md |
| `lint` | 健康检查运行记录 | make lint 结果，0 issues 或问题列表 |
| `catalog` | 完整页面目录重新生成 | `make catalog` 更新 `catalog.md` |
| `index` | （遗留别名）同 `catalog` | 旧日志兼容，新记录请用 `catalog` |
| `structural` | 结构性变更 | 新增页面类型、重构目录、添加工具脚本、升级路由 |

---

## 追加方式

**新记录写在文件顶部**（首条 `## [YYYY-MM-DD]` 之前），与 `scripts/generate_link_graph.py` 解析首页「最新知识节点」的顺序一致；不要用尾部追加。

**命令行（推荐）：**
```bash
make log OP=ingest DESC="sources/papers/xxx.md — 描述"
make log OP=lint DESC="0 issues，覆盖率 75%"
make log OP=query DESC="locomotion reward → wiki/queries/xxx.md"
```

**直接调用脚本：**
```bash
python3 scripts/append_log.py ingest "sources/papers/xxx.md — 描述"
python3 scripts/append_log.py lint "0 issues，覆盖率 75%"
```

**手动追加**（大型操作）：在 log.md **顶部**（说明行之后、现有首条 `## [` 之前）插入 `## [date] op | desc` 标题 + 详细列表。

**lint 健康报告**（`make lint` 带 `--write-log`，或 `python3 scripts/lint_wiki.py --write-log`）：同样插入顶部，标题形如 `## [YYYY-MM-DD] lint | health-check | ...`；勿在文件末尾追加。

---

## 查询方式

```bash
# 查看最近 5 条（新记录在上，用 head）
grep "^## \[" log.md | head -5

# 查看所有 ingest 操作
grep "^## \[.*\] ingest" log.md

# 查看某日期的操作
grep "^## \[2026-04-14\]" log.md

# 统计各 op 数量
grep "^## \[" log.md | grep -oP '\] \K\w+' | sort | uniq -c
```

---

## 约定

1. **只追加，不修改**：log.md 是不可变日志，已写入的条目不应被修改或删除
2. **每次操作后立即追加**：不要事后批量补写，及时性是 log 的价值所在
3. **描述要可解析**：文件路径、页面名、数字要精确，方便 LLM grep 定位
4. **无需记录微小改动**：typo 修复、格式调整不必记录；影响知识结构的操作必须记录
