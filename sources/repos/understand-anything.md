# Understand Anything（Egonex-AI/Understand-Anything）

> 来源归档（ingest）

- **标题：** Understand Anything
- **类型：** repo / agent-infrastructure / skills / knowledge-graph / developer-tools
- **组织：** [Egonex](https://github.com/Egonex-AI)（创始人 [Lum1104](https://github.com/Lum1104)）
- **代码：** <https://github.com/Egonex-AI/Understand-Anything>（**已开源**，MIT）
- **项目页 / Demo：** <https://understand-anything.com> · <https://understand-anything.com/demo/>
- **关联产品：** <https://egonex.ai>（Understand Anyone）
- **许可：** MIT
- **入库日期：** 2026-09-15
- **一句话说明：** 多平台 coding agent 插件/技能：Tree-sitter + LLM 多代理管线把代码库或 Karpathy 式 wiki 编译成交互式知识图（结构层 + 业务域视图），产出 `.ua/knowledge-graph.json` 与可共享 dashboard。

## 开源状态（步骤 2.5）

| 项 | 核查（2026-09-15） |
|----|-------------------|
| **GitHub** | 公开仓 [Egonex-AI/Understand-Anything](https://github.com/Egonex-AI/Understand-Anything)；MIT |
| **项目页** | [understand-anything.com](https://understand-anything.com) 链到仓库、Live Demo、安装说明 |
| **Demo** | [understand-anything.com/demo/](https://understand-anything.com/demo/) 可浏览器交互（无需 API key） |
| **结论** | **已开源**（插件、多代理管线、dashboard viewer、`install.sh`）。初始 `/understand` 消耗 LLM token；可用 Ollama 等本地模型。 |

## 仓库入口（README 归纳）

| 组件 | 说明 |
|------|------|
| Claude Code | `/plugin marketplace add Egonex-AI/Understand-Anything` → `/plugin install understand-anything` |
| 一键安装 | `curl -fsSL …/install.sh \| bash`（codex / opencode / openclaw / gemini / cursor / hermes / …） |
| 分析代码库 | `/understand` → `.ua/knowledge-graph.json`（遗留目录 `.understand-anything/` 仍兼容） |
| Dashboard | `/understand-dashboard` 或 `npx …/understand-anything-viewer.tgz <project>`（只读本地图，无需 LLM） |
| 业务域 | `/understand-domain` — domains / flows / steps |
| Wiki 知识库 | `/understand-knowledge ~/path/to/wiki` — Karpathy LLM wiki 模式 |
| 其它命令 | `/understand-chat`、`/understand-diff`、`/understand-explain`、`/understand-onboard` |
| 增量 | 默认只重分析变更文件；`--auto-update` 挂 post-commit hook |

## 多代理管线（Under the Hood）

| Agent | 角色 |
|-------|------|
| `project-scanner` | 发现文件、语言与框架 |
| `file-analyzer` | 函数/类/导入 → 图节点与边 |
| `architecture-analyzer` | 架构分层（API / Service / Data / UI …） |
| `tour-builder` | 引导式学习 tour |
| `graph-reviewer` | 完整性 / 引用完整性校验 |
| `domain-analyzer` | 业务域、流程、步骤（`/understand-domain`） |
| `article-analyzer` | wiki 实体、主张、隐式关系（`/understand-knowledge`） |

**混合解析：** Tree-sitter 负责确定性结构（import/call/继承）；LLM 负责摘要、标签、业务映射与 tour。

## 对 wiki 的映射

- 主实体：[Understand Anything](../../wiki/entities/understand-anything.md)
- 站点归档：[understand-anything.md](../sites/understand-anything.md)
- 交叉：[graphify](../../wiki/entities/graphify.md)、[LLM Wiki（Karpathy）](../../wiki/references/llm-wiki-karpathy.md)、[CLI-Anything](../../wiki/entities/cli-anything.md)
