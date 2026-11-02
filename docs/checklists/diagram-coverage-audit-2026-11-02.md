# 图表覆盖率审计（2026-11-02）

统计脚本：`python3 scripts/audit_diagram_coverage.py`（`--list missing-mermaid|missing-seq` 列出缺失页面）。

## 口径
- 节点 = `wiki/`、`roadmap/`、`tech-map/` 下的 `*.md`。
- 有 Mermaid = 正文含 ` ```mermaid ` 围栏（围栏写坏、从未渲染的旧图按「无」计，如 `paper-intermimicgen`、`paper-freespeed` 修复前）。
- 有源码 = frontmatter `sources:` 引用了 `sources/repos/`（含 awesome 列表类仓库页，故偏宽松）。
- 有时序图 = 含以 `sequenceDiagram` 开头的 mermaid 块。
- 「不适用」= 「源码运行时序图」标题后 300 字内含「不适用」。
- 静态文本匹配，未做 Mermaid 语法校验（语法校验见 `scripts/check_mermaid.cjs`）。

## 结果（三个时点）
| 指标 | 首次审计（当时 main） | 当前 main | 合入 [#3310](https://github.com/ImChong/Robotics_Notebooks/pull/3310) 后 |
|------|------|------|------|
| 节点总数 | 7145 | 7148 | 7148 |
| 有 Mermaid | 3171 | 3171 | **4223** |
| **无 Mermaid** | 3974 | 3977 | **2925** |
| 有源码节点 | 4106 | 4106 | 4106 |
| 有源码且有时序图 | 1041 | 1041 | 1041 |
| **有源码但无时序图** | 3065 | 3065 | **3065**（其中 411 已标「不适用」，实际待补约 2654） |
| 其中论文实体页有源码 | 2581，无时序图 1740 | 同左 | 同左（189 已标不适用） |

- 「当前 main」比首次审计多 3 个节点（main 期间新增页面），无图数随之 +3。
- 「合入 #3310 后」= 在 `claude/mermaid-pilot-papers`（#3310）分支上用同一脚本统计，并非已合入。#3310 为 1052 个页面新增流程总览（均为 `flowchart`），全仓库有图节点 +1052。
- **时序图一栏不变**：#3310 只补流程图，没有新增 `sequenceDiagram`，所以「有源码但无时序图」仍是 3065，这一缺口尚未处理。

## 无 Mermaid 分布（合入 #3310 后）
| 目录 | 节点 | 无 Mermaid | 首次审计时无 Mermaid |
|------|------|-----------|------|
| wiki/entities | 6166 | 2412 | 3462 |
| wiki/overview | 296 | 169 | 168 |
| wiki/methods | 188 | 107 | 107 |
| wiki/concepts | 240 | 101 | 101 |
| wiki/queries | 79 | 51 | 51 |
| wiki/formalizations | 48 | 33 | 33 |
| wiki/comparisons | 64 | 23 | 23 |
| wiki/tasks | 15 | 8 | 8 |
| tech-map | 20 | 20 | 20 |
| roadmap | 30 | 1 | 1 |

剩余 2925 个无图节点的构成（合入 #3310 后）：
| 类别 | 数量 | 备注 |
|------|------|------|
| `wiki/entities/paper-*` | 1415 | 其中 1041 个有 arXiv 号，可按「读论文方法部分」继续补；其余无 arXiv 号需另找一手资料 |
| 非 paper 的实体页（项目/公司/数据集等） | 997 | 其中 744 个引用 `sources/repos/`，应读官方 README 与源码 |
| 非实体页（概念/方法/总览/对比/查询等） | 492 | 需用教材或论文原文核对，不宜套模板 |
| `tech-map`、`roadmap` | 21 | 总览页，图须与下属页面一致 |

## 说明
- 补图须遵循 AGENTS.md：基于原文/源码，不得用通用模板冒充。
- #3310 的 1052 页内容为各 agent 读论文方法部分后所画，实验与附录多为略读，部分箭头是对文字描述的归纳；具体复核清单见 #3310 描述。
- 这份审计只统计「有没有图」，不评价图是否正确。
