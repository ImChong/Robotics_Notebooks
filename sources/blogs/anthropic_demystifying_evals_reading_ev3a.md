# Demystifying evals for AI agents 通读

> 来源归档（原文通读 · 支撑 `wiki/concepts/ai-agent-evaluation.md`）

- **标题：** Demystifying evals for AI agents
- **作者：** Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares, Jiri De Jonghe（Anthropic Engineering）
- **类型：** blog / agent-evaluation
- **原文：** <https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents>（2026-11 经代理抓取 HTML 全文并转文本）
- **一句话说明：** Anthropic 工程博文，系统给出 agent 评测的术语、三类 grader、能力 vs 回归 eval、各类 agent 的评测模式、pass@k / pass^k 以及从 0 到 1 的八步路线。
- **核对范围：** 通读全文：引言、评测结构与术语、为什么建 eval、三类 grader、coding / 对话 / 研究 / computer-use 四类 agent、非确定性指标、Step 0–8 路线图、与其他方法的对比表、结论、附录框架清单与致谢；两个 YAML 示例也已核对。博文引用的外部材料（SWE-bench Verified、Terminal-Bench、τ-Bench、CORE-Bench、METR 等原始论文）未逐一阅读，仅依据本文转述。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **术语：** task（含输入与成功判据）、trial、grader（可含多个 assertion）、transcript（Anthropic API 下即完整 messages 数组）、outcome（环境终态，例：订票 agent 说「已订」，但 outcome 是 SQL 库里是否真有预订记录）、eval harness、agent harness / scaffold、suite。评测「一个 agent」即评 harness 与模型的组合。
- **为何建 eval：** 无 eval 时调试是被动的，难以区分真实回归与噪声；Claude Code 先对 concision、file edits 建 eval，再扩展到 over-engineering；有 eval 的团队可在数天内升级新模型，否则需数周；eval 还给出延迟、token、成本、错误率基线。Descript 以三维（别弄坏、照我说的做、做得好）建两套 suite；Bolt 用 3 个月建系统。
- **三类 grader：** code-based（字符串匹配、fail-to-pass / pass-to-pass 测试、静态分析、工具调用与 transcript 统计；快、客观，但对合理变体脆）；model-based（rubric、自然语言断言、成对比较、多 judge 共识；灵活但非确定、需与人标定）；human（SME、抽检、A/B；金标准但贵慢）。打分可加权、二元或混合。
- **能力 vs 回归：** 能力 eval 起点低通过率；回归 eval 近 100%；高通过率的能力 eval 可「毕业」为回归套件。
- **各类 agent：** coding 以确定性测试为主（SWE-bench Verified 从约 40% 到 >80%，Terminal-Bench 做端到端任务），再辅以 transcript 质量；对话 agent 需 LLM 模拟用户（τ-Bench、τ2-Bench），多维判据（状态、轮数、语气）；研究 agent 组合 groundedness、coverage、来源质量；computer use 查终态（WebArena 的 URL / 后端状态，OSWorld 的文件 / 配置 / 数据库 / UI 属性），DOM 快但 token 多、截图慢但 token 省。另提到 Opus 4.5 在 τ2-bench 订票题中发现政策漏洞，按原题「失败」实则更优。
- **非确定性：** pass@k 随 k 升，pass^k 随 k 降；示例 75% 单次成功率、3 次全过约 42%；k=1 时相同，k=10 时 pass@k 趋近 100% 而 pass^k 趋近 0%。
- **路线图：** Step 0 20–50 条真实失败即可；Step 1 取自手工检查与 bug / support 队列；Step 2 任务无歧义 + reference solution，0% pass@100 多为坏题（Terminal-Bench 缺文件路径说明为例）；Step 3 正负平衡（Claude.ai web search 触发案例）；Step 4 隔离环境，曾见 Claude 读前序 trial 的 git 历史占便宜；Step 5 评产出不评死板路径、给部分得分、LLM 给「Unknown」出口、逐维度独立 judge、防 hack（CORE-Bench 从 42% 修到 95%，METR 阈值题配置错误）；Step 6 读 transcript；Step 7 监控饱和（Qodo 例）；Step 8 专职 evals 团队管基础设施，产品 / 领域人员贡献 task，eval-driven development。
- **组合方法：** 自动 eval、生产监控、A/B、用户反馈、人工读 transcript、系统人类研究，各有利弊，类比瑞士奶酪模型。
- **附录：** Harbor、Braintrust（autoevals）、LangSmith、Langfuse、Arize Phoenix / AX；框架只加速，质量取决于 task 与 grader。
- **自述局限（结论）：** 该领域仍在快速演进，agent 变长任务、多 agent 协作、更主观的工作会要求新技术。

## 对 wiki 的映射

- [`wiki/concepts/ai-agent-evaluation.md`](../../wiki/concepts/ai-agent-evaluation.md)
- 旧摘要归档：[`anthropic_demystifying_evals_ai_agents_2026-01-09.md`](anthropic_demystifying_evals_ai_agents_2026-01-09.md)
