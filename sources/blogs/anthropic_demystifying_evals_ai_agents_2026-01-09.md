# Demystifying evals for AI agents

> 来源归档

- **标题：** Demystifying evals for AI agents
- **类型：** blog（Anthropic Engineering）
- **作者：** Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares, Jiri De Jonghe 等
- **链接：** https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- **发布日期：** 2026-01-09
- **入库日期：** 2026-09-30
- **一句话说明：** Anthropic 工程博文：agent 评测的术语（task/trial/grader/transcript/outcome/harness/scaffold/suite）、三类 grader、能力 vs 回归 eval、按 agent 类型（coding/conversational/research/computer-use）的 proven 模式，以及从 0 到 1 的八步路线图与非确定性指标 pass@k / pass^k。
- **沉淀到 wiki：** 是 → [`wiki/concepts/ai-agent-evaluation.md`](../../wiki/concepts/ai-agent-evaluation.md)

---

## 核心术语

| 概念 | 含义 |
|------|------|
| Task | 单次测试：输入 + 成功判据 |
| Trial | 对同一 task 的一次尝试（需多次 trial 估方差） |
| Grader | 对 transcript 或 **环境终态 outcome** 打分的逻辑（可多 assertion） |
| Transcript | 完整轨迹（Anthropic API 即最终 messages 数组） |
| Outcome | 环境终态（如 DB 里是否真有订票记录） |
| Eval harness | 并发跑 task、录 transcript、聚合结果的基础设施 |
| Agent harness / scaffold | 使模型成 agent 的编排层（如 Claude Code + Agent SDK） |
| Eval suite | 测特定能力/行为的一组 task |

## 三类 Grader

- **Code-based：** 字符串匹配、单测、静态分析、工具调用检查、transcript 统计——快、客观、易脆。
- **Model-based：** rubric、成对比较、多 judge 共识——适合开放输出，需与人标定。
- **Human：** SME / 抽检 / A/B——金标准，贵。

## 能力 eval vs 回归 eval

- **Capability：** 低通过率，测「还能不能做得更好」；可随能力毕业转为回归套件。
- **Regression：** 应近 100% pass，防改坏已有行为。

## Agent 类型要点

- **Coding：** SWE-bench Verified、Terminal-Bench；优先 outcome 单测，transcript 质量为辅。
- **Conversational：** 常需 **用户模拟 LLM**；τ-Bench / τ2-Bench；多维（状态 + 轮数 + 语气 rubric）。
- **Research：**  groundedness / coverage / 来源质量；开放合成需与人标定。
- **Computer use：** WebArena / OSWorld；DOM vs 截图的 token–延迟权衡；终态 artifact 检查。

## 非确定性：pass@k vs pass^k

- **pass@k：** k 次里至少成功一次（coding 常看 pass@1）。
- **pass^k：** k 次全成功（面向用户一致性）。

## 从 0 到 1 路线图（摘要）

0. 早做：20–50 条真实失败即可，不必等数百条。  
1. 从手工发布前检查与 support/bug 队列取材。  
2. 任务无歧义 + reference solution；0% pass@100 多半是 **题/grader 坏了**。  
3. 平衡「该做/不该做」两侧（例：web search  under/over trigger）。  
4. Harness 隔离环境、防 git 历史等泄漏。  
5. 评 **产出** 多于死板工具顺序；部分得分；防 grader 被 hack。  
6. 读 transcript 验证 grader 公平性。  
7. 警惕 eval 饱和（SWE-bench 等）。  
8. 专人维护基础设施 + 产品侧贡献 task（eval-driven development）。

## 与其他信号的组合

自动化 eval、生产监控、A/B、用户反馈、人工读 transcript、系统 human study——「瑞士奶酪模型」，无单层够。

## 附录：框架

Harbor、Braintrust、LangSmith、Langfuse、Arize Phoenix/AX 等——**框架只加速基础设施，质量取决于 task 与 grader**。
