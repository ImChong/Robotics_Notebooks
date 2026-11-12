# Stanford CS146S：The Modern Software Developer（2026 秋季）一手资料核读

> 核查日期：2026-11-12。对应 [Wiki 实体页](../../wiki/entities/cs146s-modern-software-developer.md)。

- **机构：** Stanford University
- **课程官网：** <https://themodernsoftware.dev/>
- **Stanford 官方课程说明：** <https://bulletin.stanford.edu/courses/2274401>
- **课程日历：** <https://docs.google.com/document/d/1W11sWBigWIsi9wVIhhFx3wajIEnP0Qfb-ei8Ecf5VKU/edit?usp=sharing>
- **教师维护的作业仓库：** <https://github.com/mihail911/modern-software-dev-assignments>（master；README、week1–week5）
- **核对范围：** 已逐项读完 2026 秋季课程官网可公开访问的 Overview 正文、作业仓库 README.md、week1–week5 的 5 份 assignment.md 与 5 份 writeup.md 模板，以及 pyproject.toml；对照 Stanford Bulletin 核实课程身份。下述“已完成”仅指这些明确列出的公开一手材料，并非整学期全部课件、视频或未公开作业。
- **通读：** 已完成

## 课程范围与定位

官网 2026 秋季版将课程定位为 AI-native software development：MCP、agent skills、spec-driven development、loop engineering、software factory，以及面向 coding agent 的上下文、能力组织和可验证的迭代工作流。官网注明 3 units、相当于 CS111/CS161 的编程基础（推荐 CS221/229）、每周授课/动手练习/行业嘉宾与期末项目。课程强调 agent 的能力和局限，不能据此声称其实际生产力提升了固定倍数。

另有 Stanford 官方课程目录（CS146S）使用关于 Cursor、Warp、Coderabbit、Qodo 等工具的较概括描述。课程目录与 2026 官网表达侧重点不同；本文的具体作业细节以教师维护的 2026 作业仓库为准。

## 已公开的作业及可验证的任务

| 周次 | 一手原文 | 任务与可检查交付物 |
|---|---|---|
| Week 1 | [Trace Dissection of a Real Claude Code Session](https://github.com/mihail911/modern-software-dev-assignments/blob/master/week1/assignment.md) | 在自己的试验仓库中用 mitmproxy 反向代理采集 Claude Code 的 `POST /v1/messages`；分析 system prompt、tool schemas、错误恢复与计划/上下文演化；提交带证据及 OBSERVED/INFERRED 标记的 `week1/writeup.md` |
| Week 2 | [Build an MCP Server for a Third-Party API](https://github.com/mihail911/modern-software-dev-assignments/blob/master/week2/assignment.md) | 用 FastMCP 为 OAuth 2.0 API 编写至少 3 个可组合工具、至少一个写工具；支持授权码、缓存与刷新，结构化错误、写操作预览/保护、协议级测试；提交实现、示例配置、测试与报告 |
| Week 3 | [Author a Skill](https://github.com/mihail911/modern-software-dev-assignments/blob/master/week3/assignment.md) | 为真实仓库编写可复用 SKILL.md，明确触发条件和决策逻辑；测试 3 条应触发提示与 1 条不应触发的近似提示，并验证一次端到端执行 |
| Week 4 | [Make a Messy Repo Agent-Ready](https://github.com/mihail911/modern-software-dev-assignments/blob/master/week4/assignment.md) | 研究自己未编写的开源仓库，编写仓库专属 AGENTS.md 或 CLAUDE.md（架构、准确命令、约定与禁止改动范围）；以两个小任务观察实际遵循情况 |
| Week 5 | [Agent-Readiness Report](https://github.com/mihail911/modern-software-dev-assignments/blob/master/week5/assignment.md) | 对三个有差异的仓库运行 Factory Droid `/readiness-report`，比较八个维度、成熟度、失败项和改进建议；提交带原始输出和局限分析的报告 |

所有周次都有 [writeup 模板目录](https://github.com/mihail911/modern-software-dev-assignments)，其中的 `TODO` 是供学生填写的空位，**不是已完成的运行结果或已发布的参考实现**。截至核查时，该公开仓库顶层只有 week1–week5 的作业目录和环境依赖文件，不能推断后续周次没有课程内容。

## 方法与证据细节

### Week 1：运行轨迹而非凭印象解释智能体

官方要求真实的多文件任务、至少一次故障及恢复、明确计划，并对请求中 `system`、`tools`、`messages` 的实际值做证据化观察。对错误恢复、子智能体、上下文管理等结论需区分 `[OBSERVED]` 和 `[INFERRED]`。原始捕获文件不应提交；请求体可能重放密钥和敏感内容，引用前要脱敏。反向代理示例将本机 58888 端口转发到 Anthropic API，`ANTHROPIC_BASE_URL` 仅设置在临时试验仓库级别。

### Week 2：MCP 工具接口就是智能体契约

评分关注接口易用性：把枚举/范围约束写进 schema，输出筛选后再返回给 agent；错误需区分不可重试与可重试；docstring 说明 ID 来源以支持工具链；写操作添加 `dry_run` 或预览后确认。OAuth 需完整授权码交换、令牌缓存、自动刷新与最小权限，工具调用期间令牌失效时返回可操作错误，不能假设交互式弹窗。至少一个测试通过 MCP 协议层调用，而不是仅直接调用 Python 函数。

### Week 3–5：从可调用过程到可维护仓库

Week 3 的 skill 需要靠描述正确触发，遇到模糊情况有判断规则，确定步骤给精确命令；还要测试“近似但不该触发”的提示。Week 4 要把真实架构、构建/测试命令和保护边界写成可维护的仓库上下文，并从实际任务观察中修订。Week 5 以三个仓库的可比报告为证据，不可把三份报告结论外推为整个行业；Factory 的八大维度为 Style & Validation、Build System、Testing、Documentation、Dev Environment、Code Quality、Observability、Security & Governance。

## 官方仓库开放情况与局限

- **已公开：** 作业说明、报告模板、依赖说明与课程首页；README 提示 Python 3.12、Conda + Poetry，运行 `poetry install --no-interaction`。
- **未见完整参考实现：** 当前公开仓库的 week1–week5 目录各只有 `assignment.md` 与 `writeup.md`；没有课程统一的可运行 MCP server/skill 示例源代码。因此这里不绘制“源码运行时序图”，避免把作业要求误称为源码实现。
- **未核读：** 整学期全部讲义/录播、非公开作业及 Google Docs 日历的实时完整内容；原有 2026-10-30 课程日历概览可用作主题线索，不据此承诺所有讲次/日期已复核。
- **风险：** Week 1 捕获数据可能包含凭据和私人信息；Week 2 真实 OAuth 写权限可能影响在线服务。实际练习前必须使用临时仓库、测试资源、最小权限及脱敏措施。

## 与知识库的映射

- [CS146S：The Modern Software Developer](../../wiki/entities/cs146s-modern-software-developer.md)
- [Agentic Coding 软件基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)
- [规范驱动开发](../../wiki/concepts/specification-driven-development.md)
