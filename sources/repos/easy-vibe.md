# Easy-Vibe（datawhalechina/easy-vibe）

> 来源归档

- **标题：** Easy-Vibe — vibe coding 101｜The first course for AI-native product builders
- **类型：** repo / course / tutorial
- **机构：** Datawhale（数据鲸鱼）
- **链接：** <https://github.com/datawhalechina/easy-vibe>
- **站点：** <https://datawhalechina.github.io/easy-vibe/>（归档见 [`sources/sites/easy-vibe-datawhale.md`](../sites/easy-vibe-datawhale.md)）
- **Stars：** ~19.5k（2026-09-20）
- **主语言：** JavaScript（静态教程站 + 交互组件）
- **License：** CC BY-NC-SA 4.0（README badge）
- **入库日期：** 2026-09-20
- **一句话说明：** Datawhale 开源 **AI 原生产品构建** 教程：3+1 阶段（产品原型 → 全栈交付 → Claude Code/MCP/RAG/跨平台 → 附录知识库），10 语言、80+ 交互专题；含 `llms.txt` 供 OpenClaw/Cursor 等 Agent 导航。
- **为什么值得保留：** 中文/多语 **vibe coding 系统课**，与 [LearnPrompt](../repos/learnprompt.md)（任务驱动 Agent 工作台）互补；Stage 3 覆盖 **MCP、RAG、LangGraph** 与 Claude Code Skills，可交叉 [RAG 概念页](../../wiki/concepts/retrieval-augmented-generation.md) 与 [Agentic Coding 软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)。
- **沉淀到 wiki：** 是 → [`wiki/entities/easy-vibe.md`](../../wiki/entities/easy-vibe.md)

## 开源状态（步骤 2.5，2026-09-20）

- **已开源：** 全量教程 Markdown/静态站源码、交互 demo、`llms.txt`、本地 `npm` 运行脚本；GitHub Pages 在线阅读。
- **关联仓：** [hello-claw](https://github.com/datawhalechina/hello-claw) — OpenClaw 入门（README 链出，非本仓代码）。
- **结论：** 可 fork 本地预览；NC 条款限制部分商业再分发，学习/内部引用无碍。

## 核心摘录（面向 wiki 编译）

### 1) 3+1 学习路径

| 阶段 | 目标 | 典型产出 |
|------|------|----------|
| **Stage 1** | 零基础 → AI IDE → 产品原型 | 小游戏、可演示 MVP、用户访谈与迭代 |
| **Stage 2** | 全栈：前端/后端/DB/部署/支付 | 可上线 SaaS（Stripe、Supabase、Dify 等） |
| **Stage 3** | AI-Native：Claude Code、MCP、RAG、跨平台 | 小程序/Android/iOS/Electron 项目 |
| **Appendix** | 9 大领域计算机与工程素养 | 80+ 交互式原理动画（含 RAG 游戏化演示） |

### 2) Stage 3 与 Agent 栈（2026-03 起大更新）

- **Claude Code：** 安装、MCP 接 GitHub/DB/API、Skills 打包、长任务、Superpowers/TDD、移动远程开发。
- **AI 进阶：** [RAG 原理](https://datawhalechina.github.io/easy-vibe/en/stage-3/ai-advanced/rag-introduction/)、[LangGraph 高级 RAG](https://datawhalechina.github.io/easy-vibe/en/stage-3/ai-advanced/langgraph-advanced-rag/)。
- **Agent 友好：** 根目录 `llms.txt` 为 AI Agent 提供阶段决策树与目录速查（OpenClaw、Cursor、Trae 等）。

### 3) 与「vibe coding」叙事

- README 主张：**「能说话就能编程」** — 用自然语言描述需求（记账、预约、博客）再借助 AI IDE 落成产品。
- 与本库 [Agentic Coding 软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md) **互补**：Easy-Vibe 教 **入门速度与产品闭环**；该概念页强调 **取舍语言、生产可靠**，避免把 vibe coding 当能力本身。

### 4) 多语言与社区

- 教程正文 **10 语言**（zh-cn、en、zh-tw、ja、ko、es、fr、de、ar、vi）。
- **Vibe Stories**（2026-03）：真实用户（教师、学生、司机等）用 AI 做产品的叙事 carousel。

## 对 wiki 的映射

| 目标页 | 关系 |
|--------|------|
| [`wiki/entities/easy-vibe.md`](../../wiki/entities/easy-vibe.md) | 实体主链 |
| [`wiki/concepts/retrieval-augmented-generation.md`](../../wiki/concepts/retrieval-augmented-generation.md) | Stage 3 RAG 交互教程 |
| [`wiki/concepts/agentic-coding-software-fundamentals.md`](../../wiki/concepts/agentic-coding-software-fundamentals.md) | vibe coding 入门 vs SE 基础 |
| [`wiki/entities/learnprompt.md`](../../wiki/entities/learnprompt.md) | 中文 AI 实战课对照 |
| [`wiki/entities/openclaw.md`](../../wiki/entities/openclaw.md) | hello-claw 侧链 |

## 当前提炼状态

- [x] 项目页与 README 结构核查
- [x] 开源边界与 Stage 1–3 要点
- [x] wiki 实体页映射
