# AI for Auto-Research 综述通读（生命周期视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/ai-auto-research.md`）

- **标题：** AI for Auto-Research: Roadmap & User Guide
- **作者：** Lingdong Kong, Xian Sun, Wei Chow 等（Awesome AI Auto-Research Team）
- **类型：** paper / survey / research-agents
- **原文：** <https://arxiv.org/abs/2605.18661>（2026-11 通过 arXiv PDF v2，2026-07-20 版获取全文）
- **一句话说明：** 按 4 阶段 8 子阶段梳理 AI 在学术研究全生命周期的工具、基准与方法族，核心结论是「产物生成快于科学验证」，可信部署形态是人机共治。
- **核对范围：** 通读摘要、§1 引言、§2 预备（生命周期、五类方法族、文献收集范围、时间线、表 2 基准）、§3–§6 全部四个阶段（含各阶段「Findings and Observations」）、§7 跨阶段分析（端到端系统、评测、五条洞见、开放挑战）、§8 结论与「Responsible Use」；附录 B（与五篇相关综述的覆盖对比）。附录 A 的 A.1–A.5 工具清单表（逐条罗列已在正文引用的系统）与参考文献列表仅浏览结构，未逐条核对；各系统的原论文未逐一阅读，下列数字均为综述转述。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **框架（§2.1）：** 4 个认识论阶段 / 8 阶段：Creation（S1 Idea、S2 Lit. Review、S3 Coding & Exp.、S4 Tables & Figures）、Writing（S5）、Validation（S6 Peer Review、S7 Rebuttal & Revision）、Dissemination（S8 Paper2X）。时间序为主但非线性：审稿可回到 S3 补实验，Paper2X 可暴露 S5 的问题。
- **方法族（§2.2）：** prompt、RAG、training-free agentic、training-based、hybrid；混合系统日益占主导。
- **五项核心发现（§1）：** 结构化且可外部核验的任务 AI 强，开放新颖任务骤降；产物生成快于验证；人机共治最可靠；有效系统采用探索-执行-验证分层；AI 使用变成治理问题而非检测问题。
- **S1（§3.1）：** AI 想法实施后质量下降（Δ=−1.98，人类想法 −0.63）；IdeaBench 上新颖性 >0.6 而可行性 <0.5；LLM-as-Judge 的新颖度评分与后续影响呈负相关（ρ=−0.29）；想法多样性塌缩被认为是结构性的。
- **S2（§3.2）：** 最快成熟的阶段；单遍 → 结构感知 → 多 agent → 编辑器集成四代；ScholarCopilot 引文 top-1 准确率仅 40.1%；幻觉从明显捏造转向「有引用但不被支持」的细微错位。
- **S3（§3.3）：** SWE-bench Verified 超过 76%，但 SWE-bench Pro 23%、SWE-EVO 25%；新颖 ML 实现 ResearchCodeBench 最佳 37.3%（58.6% 错误为语义错误：代码能跑但实现了错误算法）、SciReplicate-Bench 约 39%；闭环系统约 12 次实验/小时；FunSearch、AlphaEvolve、AIDE 说明「生成+搜索+评估」优于单次生成；引用 MLR-Bench 称全自主结果中约 80% 为捏造。
- **S4（§3.4）：** 数据可视化 90%+ 执行通过；框架图、消融表、公式/TikZ 精度悬崖（公式准确率随复杂度由 78.8% 降至 15%）；「看起来对不等于科学上对」。
- **S5（§4）：** CS 摘要中可检测到 AI 修改最高 17.5%；CycleResearcher 生成论文 ICLR 评分 5.36（录用均值 5.69）；差距在论证深度而非流畅度；检测器误报高，会议转向「声明」政策。
- **S6（§5.1）：** LLM 审稿分数偏高（AI 6.86 vs 人 5.70，把 95.8% 被拒稿件判为可接收）；ICLR 2025 随机实验显示 LLM 反馈改进人类评审，89% 提升了评审质量；提示注入、良性形容词触发等可操纵评分；检测 AI 评审不可靠。
- **S7（§5.2）：** ICLR 2024–2025 约 75–81% 分数在 rebuttal 后不变；ICLR 2025 作者平均每篇 11.8 条承诺，约 25% 在终稿未兑现；现有系统不能自动补新实验（S7→S3 回路未自动化）。
- **S8（§6）：** 海报约 $0.005、token 减少 87%；瓶颈是信任与保真而非成本；Paper2Agent 把论文+代码转为 MCP 服务，需同时评估沟通与可复现性。
- **端到端四类架构（§7.1）：** 顺序流水线、搜索/自改进、技能/工具集成、多 agent/社区；共同决定因素是阶段交接处是否有可追溯状态与验证；多数系统覆盖 Creation 与 Writing，Validation 稀少，Dissemination 基本缺位。
- **评测（§7.2）：** 专家评审、LLM/Agent-as-Judge、自动指标、执行落地评测、过程/轨迹评测五类；缺口含新颖性定义、污染、跨系统不可比、跨领域未测、成本、尚无覆盖全生命周期的基准。
- **五条跨阶段洞见（§7.3）：** 见上述核心发现；多 agent 只在可分解/并行任务上增益，在顺序推理上可能因协调开销与错误传播退化。
- **开放挑战（§7.4）：** 跨阶段忠实性、科学判断与新颖性评估、验证与可复现、引用版本与来源溯源、披露与治理、跨领域与基础设施可及、人类专长与认知所有权；提出四条设计原则：全生命周期溯源、执行/检索落地、阶段边界设人类检查点、透明披露 AI 参与。

## 对 wiki 的映射

- [`wiki/concepts/ai-auto-research.md`](../../wiki/concepts/ai-auto-research.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2605.18661>
- 既有摘要归档：[ai_auto_research_survey_2605_18661.md](ai_auto_research_survey_2605_18661.md)
