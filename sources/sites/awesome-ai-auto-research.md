# Awesome AI Auto-Research（GitHub Pages）

- **类型**：项目静态站点 / 综述可视化导航
- **收录日期**：2026-06-20
- **站点**：<https://worldbench.github.io/awesome-ai-auto-research>
- **同源仓库**：<https://github.com/worldbench/awesome-ai-auto-research>
- **综述论文**：<https://arxiv.org/abs/2605.18661>

## 一句话

把 **AI 辅助学术研究全生命周期** 综述（四阶段八阶段、五项核心发现、工具与基准 inventory）整理成 **可扫读的章节化网页**，并挂载可搜索论文库（与仓库同步）。

## 为什么值得保留

- **边界叙事**清晰：核心挑战从「能否生成研究形态」转为「能否保留证据、判断、溯源与问责」。
- 各 stage 页面给出 **子主题标签 + 代表方法**（如 S3 的 PaperCoder / AIDE / R&D-Agent；S6 的 DeepReviewer / MARG），适合 wiki 写「延伸阅读」时的 **外部索引**，避免在 wiki 内复制长列表。
- 强调 **stage-dependent reliability**：结构化检索型任务成熟快；新颖 idea、研究级代码与科学判断仍不可靠。

## 站点摘录（2026-06-20 抓取要点）

来源：<https://worldbench.github.io/awesome-ai-auto-research>

- **四阶段**：Creation（ideation · literature · code & exp. · tables & figures）→ Writing → Validation（peer review · rebuttal）→ Dissemination（Paper2X 全家桶）。
- **五项发现（站点首页）**：能力随任务结构化程度变化 · 生成快于验证 · 人机共治最可信 · 分层架构（explore / execute / verify）· 治理问题。
- **S1 Idea**：新颖性在实施后常退化；LLM-as-Judge 可能与真实影响力负相关。
- **S2 Literature**：增长最快；引用 top-1 精度仍低（ScholarCopilot ~40.1% 量级）。
- **S3 Coding**：SWE-bench 高 ≠ 研究代码就绪；ResearchCodeBench 等显示 **语义错误** 为主。
- **S6 Peer Review**：单独 AI 审稿不安全（分数膨胀、对抗脆弱）；**AI 辅助改人类审稿** 有随机对照证据。
- **S8 Paper2X**：信任瓶颈在 **忠实度** — 公众材料可能夸大或省略局限。

## 对 wiki 的映射

- 主沉淀：[AI Auto-Research（学术研究自动化）](../../wiki/concepts/ai-auto-research.md)
- 原始论文档：[ai_auto_research_survey_2605_18661.md](../papers/ai_auto_research_survey_2605_18661.md)
- 列表维护入口：[awesome-ai-auto-research.md](../repos/awesome-ai-auto-research.md)
