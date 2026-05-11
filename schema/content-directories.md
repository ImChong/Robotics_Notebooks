# 内容目录怎么选

> 精炼版：新资料、新页面、新路线应落在哪个根目录。详细维护流程仍以 [ingest-workflow.md](ingest-workflow.md) 为准。

---

## 一句话对照

| 目录 | 一句话 |
|------|--------|
| [`wiki/`](../wiki/) | 已提炼的**结构化知识**（概念、方法、任务、实体、对比、形式化）。读者在此「理解 + 跳转」。 |
| [`sources/`](../sources/) | **原始资料输入**：论文摘录、仓库导航、博客笔记等；ingest 第一站，不替代 wiki 正文。 |
| [`references/`](../references/) | **按主题索引的深挖入口**（论文列表、repo 清单、benchmark）；在已懂概念后继续扩展阅读。 |
| [`resources/`](../resources/) | **资源沉淀层**：课程笔记、训练计划、长笔记等，不追求与全站 wiki 同粒度互链。 |
| [`roadmap/`](../roadmap/) | **主学习路径**与条件分支路线；首页与读者默认入口。 |
| [`wiki/roadmaps/`](../wiki/roadmaps/) | 与 wiki 主题**强绑定**的专题路线（见 ingest 步骤 4 后的说明）。 |
| [`tech-map/`](../tech-map/) | **技术地图**模块依赖与栈全景，与路线页互补。 |
| [`exports/`](../exports/) | **脚本生成**的 JSON 等；勿手改。站点可读副本见 `docs/exports/`（见 [contributing-ci.md](../docs/contributing-ci.md)）。 |

---

## 决策漏斗（维护时用）

1. **是否仍是「原材料」**（链接堆砌、摘录、未形成可引用知识段落）→ 是则进 **`sources/`**（或先收进 `resources/` 再择机提炼）。
2. **是否已是可交叉引用的知识页**（有定义、机制、误区、参考来源）→ 是则进 **`wiki/`** 下对应子目录（概念 / 方法 / 任务 / …）。
3. **是否「已知概念、只要按主题找论文或 repo」**→ **`references/`**。
4. **是否长材料、课程、个人训练笔记、暂不进 wiki 的沉淀**→ **`resources/`**。
5. **是否读者要跟着走的路线**→ 默认 **`roadmap/`**；仅当与某一 wiki 子树强耦合且不必占根级主入口时考虑 **`wiki/roadmaps/`**。

---

## 常见混淆

- **`sources/` vs `references/`**：前者是「本条资料的归档与 ingest 溯源」；后者是「按研究方向整理的阅读索引」。一篇论文可先出现在 `sources/papers/`，再在 `references/papers/` 里被归类收录。
- **`wiki/` vs `resources/`**：能进 wiki 的应尽量满足 frontmatter、参考来源与互链规范；做不到时先放 `resources/`，避免把 wiki 变成笔记堆。
- **手改 `exports/` 或 `docs/exports/`**：不要；用 `make ci-preflight` 等命令再生。

---

## 与其它规范的关系

- Schema 总索引：[README.md](README.md)
- Ingest 操作步骤：[ingest-workflow.md](ingest-workflow.md)
- 内链写法：[linking.md](linking.md)
- 仓库总览：[README.md](../README.md) 中「项目结构」表
