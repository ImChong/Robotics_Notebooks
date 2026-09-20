# LearnPrompt 教程站 — learnprompt.pro

- **类型**：教程站 / AI Practice Wiki（原始资料归档）
- **收录日期**：2026-09-20
- **主链接**：<https://www.learnprompt.pro/>
- **英文站**：<https://www.learnprompt.pro/en/>
- **旧版归档**：<https://v1.learnprompt.pro/>（早期 AIGC 工具入门课程）
- **代码：** <https://github.com/LearnPrompt/LearnPrompt>
- **抓取说明**：以 **2026-09-20** 对首页与 sitemap 公开 HTML 为准；站点自述为 Astro 静态站，含 Showcase 与独立审稿流程。

## 一句话

永久免费开源的中文 **AI native 实战教程站**：围绕真实任务（写稿、做 PPT、改代码、搭知识库、维护 Agent 工作台）组织 **8 条学习路径、47 篇可重放教程**，覆盖 Claude Code、Codex、Agent Harness、Skills、Loop、Obsidian 与 Hermes/OpenClaw；由「卡尔的 AI 沃茨」/ GoodCase.ai 维护。

## 为什么值得保留

- **与本库维护范式同向：** 新版站点强调 **Idea → Task → Delivery → Memory**，Obsidian/Markdown 作 Agent 记忆、Skills 沉淀重复流程 — 与 [Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md) 及本仓库 [ingest/query/lint](../../schema/ingest-workflow.md) 文化直接相交。
- **Coding Agent 中文路径稀缺：** 8 条路径中 **Claude Code / Codex / Agent Skills / Loop** 四块，是本站在 `AGENTS.md`、Cloud Agent、skills 生态下的 **读者友好中文入口**。
- **OpenClaw / Hermes 教程锚点：** 含 [OpenClaw 架构导读](https://www.learnprompt.pro/agent-frameworks/openclaw-architecture-guide/) 等，可补 [OpenClaw 实体页](../../wiki/entities/openclaw.md) 的 **上手文档** 侧链。
- **Skill 工坊生态：** 站点 `/skills/` 与 GitHub 组织下 **鲁班、庖丁、蔡伦、阿福、愚公、搭子** 等 skills 互链，可与 [mattpocock/skills](../../wiki/entities/mattpocock-skills.md)、[Nuwa Skill](../../wiki/entities/nuwa-skill.md) 对照阅读。

## 站点结构（2026-09-20 首页）

| 指标 | 内容 |
|------|------|
| 教程篇数 | 47 篇（每篇含来源、教学图、Showcase、独立审稿） |
| 学习路径 | 8 条（可按任务选入口，不要求顺序通读） |
| 质量主张 | 正常路径 + 失败场景 + 越界反例均跑 Showcase；命令/权限回官方资料核验并标注日期 |

### 八条路径（Track）

| # | 路径 | 入口 | 规模 | 焦点 |
|---|------|------|------|------|
| 01 | AI 编程入门 | `/ai-coding/` | 5 篇 | 任务卡 → diff → 验收的第一轮交付 |
| 02 | Claude Code | `/claude-code/` | 8 篇 | 本地项目、规则、长会话、Skills、多 Agent |
| 03 | Codex | `/codex/` | 6 篇 | CLI / IDE / 桌面 / Cloud 四执行面与审查 |
| 04 | Agent 工程 | `/agent-engineering/` | 6 篇 | Harness 五组件：指令、约束、记忆、反馈、编排 |
| 05 | Agent Skills | `/agent-skills/` | 7 篇 | 第一个 `SKILL.md`、触发条件与验收 |
| 06 | Loop Engineering | `/loop-engineering/` | 1 篇 | 持续循环五动作（状态、验收、下一步） |
| 07 | Obsidian AI | `/obsidian-ai/` | 5 篇 | Markdown 目录、索引、项目交接作 Agent 记忆 |
| 08 | Hermes / OpenClaw | `/agent-frameworks/` | 3 篇 | 长驻 Agent 架构、消息流与成本 |

### 每条路径精选一篇（首页 Featured）

1. [从自然语言需求到可运行 MVP](https://www.learnprompt.pro/ai-coding/natural-language-to-mvp/)
2. [Claude Code 安装与第一个项目](https://www.learnprompt.pro/claude-code/install-and-first-project/)
3. [Codex 的四个执行面怎么选](https://www.learnprompt.pro/codex/codex-form-factors/)
4. [Harness 的五个组件](https://www.learnprompt.pro/agent-engineering/what-is-harness/)
5. [第一个 SKILL.md 怎么写](https://www.learnprompt.pro/agent-skills/first-skill-md/)
6. [一个 Loop 的五个动作](https://www.learnprompt.pro/loop-engineering/five-moves/)
7. [Markdown 何时才算 Agent 记忆](https://www.learnprompt.pro/obsidian-ai/markdown-as-agent-memory/)
8. [OpenClaw 架构导读](https://www.learnprompt.pro/agent-frameworks/openclaw-architecture-guide/)

## 开源状态

**已开源**：教程站源码与内容维护于 [LearnPrompt/LearnPrompt](https://github.com/LearnPrompt/LearnPrompt)（GitHub 组织默认 Profile README 兼项目仓）；关联 skills 与子项目分散于 `LearnPrompt/*` 各独立仓库（如 [luban-skill](https://github.com/LearnPrompt/luban-skill)、[carl-skills](https://github.com/LearnPrompt/carl-skills)、[andrej-karpathy-skills](https://github.com/LearnPrompt/andrej-karpathy-skills)）。

## 关联项目（组织生态，非本站逐条 ingest）

| 项目 | 说明 |
|------|------|
| [ai-news-radar](https://github.com/LearnPrompt/ai-news-radar) | AI 信息雷达 / 伯乐 Skill |
| [luban-skill](https://github.com/LearnPrompt/luban-skill) | 把可用 Skill 打磨成可安装公共资产 |
| [humanize-ppt](https://github.com/LearnPrompt/humanize-ppt) | 先叙事主线再生成 PPT |
| [skillrush-town](https://github.com/LearnPrompt/skillrush-town) | Skill 排行榜快照与历史对比 |
| [andrej-karpathy-skills](https://github.com/LearnPrompt/andrej-karpathy-skills) | Karpathy 公开方法论 → 14 个 Agent Skills |
| [carl-skills](https://github.com/LearnPrompt/carl-skills) | 作者日常 AI 工作流 skills 沉淀仓 |

## 沉淀到 wiki

是 → [`wiki/entities/learnprompt.md`](../wiki/entities/learnprompt.md)
