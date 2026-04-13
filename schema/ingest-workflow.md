# Wiki Operations

> 本文件定义 Robotics_Notebooks 知识库的日常维护操作规范。
> 基于 Karpathy LLM Wiki 模式：LLM 是维护者，人类是 curator。
> 三种核心 ops：Ingest（吃进）、Query（查询）、Lint（体检）。

---

## Op 1：Ingest — 新资料进入知识库

### 触发条件

满足以下任一条时，执行 ingest：
- 读了一篇值得留存的论文 / 博客 / 教程
- 看了有价值的视频或课程笔记
- 发现了新的工具、仓库、数据集
- 对某个概念有了新的理解或修正

### 步骤 1：判断资料类型

- `paper` — 论文
- `blog` — 博客文章
- `course` — 课程 / 教程
- `video` — 视频
- `repo` — 代码仓库
- `personal` — 个人理解 / 总结

### 步骤 2：进入 `sources/`

原始资料先收录到 `sources/` 对应位置，保留来源信息。

建议记录：
- 标题、链接、类型
- 一句话说明
- 为什么值得保留
- 来源日期（如果知道）

### 步骤 3：判断是否沉淀到 `wiki/`

只有满足以下至少一条，才升格进 `wiki/`：
- 能解释一个重要概念
- 能补全一个方法页
- 能帮助做选型对比
- 能影响学习路线或研究判断

### 步骤 4：更新相关页面

若资料有价值：
- 更新一个已有 wiki 页（补充 cross-reference、关联关系）
- 或新建一个 wiki 页

同时补充：
- `related` — 关联页面
- `继续深挖入口` — references 链接
- 必要时更新 `index.md`

### 步骤 5：更新导出数据

如果新增或大幅修改了 wiki 页面，运行导出脚本同步到前端：

```bash
python3 scripts/export_minimal.py
```

### 步骤 6：记录到 `log.md`

每次 ingest 都追加到 `log.md`，格式：

```markdown
## [YYYY-MM-DD] ingest | <type> | <title>

- <简短说明做了什么事>
- <影响的其他页面>
- <任何值得留存的结论>
```

---

## Op 2：Query — 针对知识库提问

### 触发条件

- 想了解某个概念但不想翻完所有原始资料
- 想找"哪个页面涉及 X 和 Y 的对比"
- 想确认"关于这个主题目前 wiki 里有什么"

### 执行方式

直接在对话里向 LLM 描述你的问题，附上 `index.md` 的内容作为上下文。

### Query 结果写回 wiki

好的 Query 结果不应该消失在聊天记录里。如果 LLM 给出了：
- 一个有价值的对比分析 → 写成 wiki comparison 页面
- 一个新的联系或洞见 → 补充到相关概念页的 `related` 或正文
- 一段值得留存的总结 → 写成 wiki 概念页

写回后，在 `log.md` 追加一条：

```markdown
## [YYYY-MM-DD] query | <topic> | <question summary>

- Q: <问题摘要>
- A: <结论摘要>
- 写回：<写到哪个页面了>
```

---

## Op 3：Lint — 定期健康检查

### 触发条件

- 每隔一段时间（建议至少每 1-2 周一次）
- 新增大量页面后
- 发现某块内容明显过时或矛盾时

### 检查清单

1. **Orphan pages**：是否有页面没有任何 inbound link
2. **矛盾点**：新旧页面之间是否存在事实冲突
3. **缺失的 cross-reference**：某个概念被提到但没有自己的页面
4. **过时内容**：某页面长期未更新但相关来源已有新发展
5. **空壳页面**：只有标题没有实质内容的页面
6. **stale related**：某页面的 `related` 列表已经明显不反映当前知识结构

### Lint 后记录

```markdown
## [YYYY-MM-DD] lint | health-check | <简短描述>

- 发现：<问题 1>
- 发现：<问题 2>
- 修复：<问题 X 已修复>
```

---

## 操作符约定

| Op | 含义 |
|----|------|
| `ingest` | 新资料进入知识库 |
| `query` | 向知识库提问并将结果写回 |
| `lint` | 定期健康检查 |
| `structural` | 结构性变更（新增页面类型、路由调整等） |

---

## 与 `docs/change-log.md` 的分工

- **`log.md`** = 每次操作的时间线（ingest / query / lint / structural）
- **`docs/change-log.md`** = 重要结构性变更的里程碑记录

日常维护优先写 `log.md`；里程碑性质的变化（如 V1 → V2）才写 `docs/change-log.md`。

---

## 注意事项

- 不要把 source 直接复制成 wiki — 要提炼，不是转存
- 不要把 wiki 页写成纯外链列表 — 要有知识归纳
- 不要为了收集而收集 — 优先服务学习与研究主线
- 不要在 ingest 时一次性做太多事 — 一次一条资料，深度到位再推进
