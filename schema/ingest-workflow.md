# Wiki Operations

> 本文件定义 Robotics_Notebooks 知识库的日常维护操作规范。
> 基于 Karpathy LLM Wiki 模式：LLM 是维护者，人类是 curator。
> 四种核心 ops：Ingest（吃进）、Query（查询）、Lint（体检）、Index（索引）。

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

### 步骤 1.5：使用 `ingest_paper.py` 快速生成模板

对于论文类资料，优先使用工具生成 sources/papers/ 模板，避免手动格式出错：

```bash
# 生成模板（填写后手动编辑论文摘录）
make ingest NAME=my_topic TITLE="论文集合标题" DESC="一句话说明"

# 等价命令
python3 scripts/ingest_paper.py my_topic --title "..." --desc "..."
```

模板生成后，编辑文件填写核心论文摘录（至少 3 条），并在每条下方填写 `对 wiki 的映射`。

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

### 步骤 4：判断放进哪个 wiki 子目录

| 资料性质 | 目标位置 |
|---------|---------|
| 形式化基础（数学/理论框架）| `wiki/formalizations/` |
| 核心概念/方法 | `wiki/concepts/` 或 `wiki/methods/` |
| 实体（工具/框架/硬件）| `wiki/entities/` |
| 任务方向 | `wiki/tasks/` |
| 对比分析 | `wiki/comparisons/` |
| 学习路线 | `roadmap/` 或 `wiki/roadmaps/` |

### 步骤 5：更新相关页面

若资料有价值：
- 更新一个已有 wiki 页（补充 cross-reference、关联关系）
- 或新建一个 wiki 页

同时补充：
- `参考来源` — 必须标注本次 ingest 的原始资料（至少 1 条），格式：
  ```markdown
  ## 参考来源
  - [资料标题](../sources/xxx.md)
  ```
- `关联页面` — 至少 2 个相关 wiki 页
- `推荐继续阅读` — 至少 1 个外部资源
- **Mermaid 流程图（推荐，管线类资料建议必做）** — 若资料的主贡献是**多阶段数据流、训练流水线或闭环系统**（例如「采集 → 重定向 → 仿真修正 → 策略训练」），在升格后的 wiki 页中增加一节（如「流程总览」），用 ```mermaid 代码块绘制**一张主干流程图**：节点对应模块边界，边对应数据/监督信号流向；子细节可用文字分节或第二张图，避免单图过度拥挤。渲染侧以 GitHub / 站点 Mermaid 为准，避免使用非标准语法。
- 必要时更新 `index.md`

> **为什么要在 wiki 页面内标注来源**：log.md 记录了操作时间线，但页面本身也应能追溯知识来源，
> 这样读者不依赖 log.md 就能知道这个 wiki 页的知识是从哪里编译来的。

### 步骤 6：更新 index.md Page Catalog

每次新增 wiki 页面后，必须重新生成 Page Catalog 更新 index.md：

```bash
make catalog  # 等价于 python3 scripts/generate_page_catalog.py
```

### 步骤 7：运行导出脚本

如果新增或大幅修改了 wiki 页面，运行导出脚本同步到前端：

```bash
make export  # 等价于 python3 scripts/export_minimal.py
# 同时更新 exports/index-v1.json、exports/site-data-v1.json、docs/sitemap.xml
```

### 步骤 8：健康检查

确认知识库健康状态：

```bash
make lint  # 目标：0 issues
# 输出包含 Sources 覆盖率报告
```

### 步骤 9：记录到 `log.md`

每次 ingest 都追加到 `log.md`：

```bash
make log OP=ingest DESC="sources/papers/xxx.md — 简述覆盖的 wiki 页面"
# 等价于 python3 scripts/append_log.py ingest "sources/papers/xxx.md — 简述"
```

格式参考 `schema/log-format.md`。

---

## Op 2：Query — 针对知识库提问

### 触发条件

- 想了解某个概念但不想翻完所有原始资料
- 想找"哪个页面涉及 X 和 Y 的对比"
- 想确认"关于这个主题目前 wiki 里有什么"
- 主动发现了一个值得深挖的连接

### 标准流程

**Step 1：定位相关页面**

先读 `index.md` 和相关分类页，找到最接近的 2-4 个 wiki 页。

**Step 2：综合分析**

精读相关页面，提取：
- 各页的核心结论
- 它们的相同点和分歧点
- 是否有页面已经过时或相互矛盾

**Step 3：形成答案**

好的答案可以是：
- 一个 wiki comparison 页面（两个方法的系统对比）
- 一个新的概念页（把多个来源的洞见综合成一篇）
- 对已有页面的补充（补充 cross-reference 或纠正矛盾）

**Step 4：判断是否写回 wiki**

写回标准：有独立的认知价值——不是简单的复述，而是有新的连接、对比或洞见。

**Step 5：写回到 wiki**

- 独立洞见（跨多个 wiki 页面的综合分析）→ 新建 `wiki/queries/xxx.md`，并在页面顶部注明触发问题
- 对比分析 → 新建 `wiki/comparisons/xxx.md`
- 新概念 → 新建 `wiki/concepts/xxx.md`
- 补充性内容 → 找到最接近的已有页面插入

写回页面格式要求（query 产物特有）：
```markdown
> **Query 产物**：本页由以下问题触发：「<问题一句话>」
> 综合来源：<列出精读的 wiki 页面>
```

**Step 6：记录到 `log.md`**

```markdown
## [YYYY-MM-DD] query | <topic> | <问题一句话>

- Q: <问题摘要>
- A: <结论摘要>
- 写回：<写到哪个页面了，新页面还是已有页面>
- 涉及页面：<精读了哪些 wiki 页>
```

---

## Op 3：Lint — 定期健康检查

### 触发条件

- 每天早上定时任务自动跑（主要）
- 每隔一段时间（建议至少每 1-2 周一次）手动触发
- 新增大量页面后
- 发现某块内容明显过时或矛盾时

### 检查清单

1. **Orphan pages**：是否有页面没有任何 inbound link
2. **矛盾点**：新旧页面之间是否存在事实冲突
3. **缺失的 cross-reference**：某个概念被提到但没有自己的页面
4. **过时内容**：某页面长期未更新但相关来源已有新发展
5. **空壳页面**：只有标题没有实质内容的页面
6. **stale related**：某页面的 `related` 列表已经明显不反映当前知识结构
7. **索引一致性**：index.md 中的页面列表是否和实际 wiki 目录一致

### Lint 后记录

```markdown
## [YYYY-MM-DD] lint | health-check | <简短描述>

- 发现：<问题 1>
- 发现：<问题 2>
- 修复：<问题 X 已修复>
- 待处理：<问题 Y 暂未处理，原因>
```

---

## Op 4：Index — 维护 index.md

### 触发条件

- 每次新增 wiki 页面后
- 每次删除或重命名 wiki 页面后
- 定期检查（建议配合 Lint 一起做）

### 维护原则

`index.md` 是知识库的总入口，每次新增页面必须同步更新，不允许页面存在于 wiki 目录但不在 index.md 中。

更新内容：
- 在对应的 `### Wiki Pages` / `### Formalizations` / `### Comparisons` 等区块下追加新页面
- 格式：`[<页面名>](<路径>) — <一句话摘要>`
- 一句话摘要从页面顶部的"**一句话定义**"字段提取

---

## 操作符约定

| Op | 含义 |
|----|------|
| `ingest` | 新资料进入知识库 |
| `query` | 向知识库提问并将结果写回 |
| `lint` | 定期健康检查 |
| `index` | 维护 index.md 索引 |
| `structural` | 结构性变更（新增页面类型、路由调整等） |

---

## 与 `docs/change-log.md` 的分工

- **`log.md`** = 每次操作的时间线（ingest / query / lint / structural）
- **`docs/change-log.md`** = 重要结构性变更的里程碑记录（V1→V2、新入口/路由重大调整等）

日常维护优先写 `log.md`；里程碑性质的变化（如版本升级、重大架构调整）才写 `docs/change-log.md`。

---

## 注意事项

- 不要把 source 直接复制成 wiki — 要提炼，不是转存
- 不要把 wiki 页写成纯外链列表 — 要有知识归纳
- 不要为了收集而收集 — 优先服务学习与研究主线
- 不要在 ingest 时一次性做太多事 — 一次一条资料，深度到位再推进
- 每次 ingest 都要更新 index.md 和 log.md，不要遗漏
