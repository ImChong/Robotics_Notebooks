# 图表覆盖率审计（2026-11-02）

统计脚本：`python3 scripts/audit_diagram_coverage.py`（`--list missing-mermaid|missing-seq` 列出缺失页面）。

## 口径
- 节点 = `wiki/`、`roadmap/`、`tech-map/` 下的 `*.md`（共 7145）。
- 有源码 = frontmatter `sources:` 引用了 `sources/repos/`（含 awesome 列表类仓库页，故偏宽松）。
- 有时序图 = 含以 `sequenceDiagram` 开头的 mermaid 块。
- 「不适用」= 「源码运行时序图」标题后 300 字内含「不适用」。

## 结果
| 指标 | 数量 |
|------|------|
| 节点总数 | 7145 |
| 有 Mermaid | 3171 |
| **无 Mermaid** | **3974** |
| 有源码节点 | 4106 |
| 有源码且有时序图 | 1041 |
| **有源码但无时序图** | **3065**（其中 411 已标「不适用」，实际待补约 2654） |
| 其中论文实体页（`paper-*`）有源码 | 2581，无时序图 1740（189 已标不适用） |

## 无 Mermaid 分布
| 目录 | 节点 | 无 Mermaid |
|------|------|-----------|
| wiki/entities | 6164 | 3462 |
| wiki/overview | 295 | 168 |
| wiki/methods | 188 | 107 |
| wiki/concepts | 240 | 101 |
| wiki/queries | 79 | 51 |
| wiki/formalizations | 48 | 33 |
| wiki/comparisons | 64 | 23 |
| wiki/tasks | 15 | 8 |
| tech-map | 20 | 20 |
| roadmap | 30 | 1 |

## 说明
- 判定为静态文本匹配，未做 Mermaid 语法校验（语法校验见 `scripts/check_mermaid.cjs`）。
- 补图须遵循 AGENTS.md：基于原文/源码，不得用通用模板冒充。
