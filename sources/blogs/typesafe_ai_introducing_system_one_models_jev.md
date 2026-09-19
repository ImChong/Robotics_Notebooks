# typesafe_ai_introducing_system_one_models_jev

> 来源归档（blog / 官方）

- **标题：** Introducing System One Models & Jev
- **类型：** blog
- **作者：** Diogo Almeida（TypeSafe AI 创始人；前 OpenAI，ChatGPT 指令跟随研究相关）
- **原始链接：** https://typesafe.ai/blog/introducing-system-one-models-and-jev
- **发表日期：** 2026-09-15
- **入库日期：** 2026-09-19
- **抓取方式：** WebFetch（typesafe.ai）
- **一句话说明：** 发布 **System One Models** 新模型类与首个公开模型 **Jev**——面向软件自动化的 **并行采样、类型安全、校准概率** 结构化决策，训练算法 **RLCD**（Reinforcement Learning for Calibrated Decisions）；相对 LLM 在 System One 任务上宣称 **40–200× 延迟**、**两个数量级** 成本优势。
- **沉淀到 wiki：** [`wiki/entities/typesafe-jev.md`](../../wiki/entities/typesafe-jev.md)

## 核心摘录（归纳，非全文）

### System One Models vs LLM

| 维度 | 传统 LLM | System One + Jev |
|------|----------|------------------|
| 优化 | RLHF / RLVR | **RLCD**（校准决策） |
| 输入 | 文本/消息序列 | **结构化 program state** |
| 输出 | 字符串（需 parse/validate） | **预定义 schema 的类型安全值 + 概率/置信度** |
| 采样 | 自回归逐 token | **并行** 单次 query 输出全部字段 |
| 典型延迟 | 3–329 s（前沿 LLM） | **70–500 ms** |
| 定价（文内） | 输入 $0.20–$10/MTok；输出 ~5× | 输入 **$0.042/MTok**；**输出免费** |
| 幻觉/类型 | 仍可能幻觉与类型错误 | **schema 匹配保证 0% 类型错误**；校准置信度 |

### Jev 定位

> 「frontier-intelligence function call」：非结构化 state 进，**typed probabilistic decisions** 出。

**适用（文内）：** AI 工作流 / smart if-statement、分类路由评分、大数据 map-reduce 特征化、**实时应用**（~100 ms）、guardrail/越狱检测/LLM 输出评分验证。

**不适用：** 开放式聊天、需要长文本生成的 demo、无 schema 的创意写作。

### 证据与 demo

- **Side-by-side demo：** 并行输出全部概率 vs GPT-5.6 Terra 自回归。
- **Workflow evals：** 固定代码 workflow，以 GPT-6 Astra + Fable 5.1 平均为参考概率；Jev 在成本-智能 Pareto 前沿。
- **首页宣称（workflow 基准）：** **193.6× faster，444.6× cheaper**（文内称偏乐观上界）。
- **Fun demos：** Doom 实时 bot（~10 QPS）、Wikiracing 高基数链接选择（cardinality ≤255）。

### 命名

- **System One：** 来自 Kahneman *Thinking, Fast and Slow* 的 System 1；强调快速结构化决策。
- **Jev：** William Stanley Jevons（效率提升→需求扩大，类比智能成本下降解锁用例）。

## 对 wiki 的映射

- [typesafe-jev](../../wiki/entities/typesafe-jev.md)（主实体）
- [behavior-tree-vla-orchestration](../../wiki/concepts/behavior-tree-vla-orchestration.md)（代码内 fuzzy 分支 / 路由）
- [deepseek-harness](../../wiki/entities/deepseek-harness.md)（字符串 agent vs 决策 API 分层）

## 可信度与使用边界

- 性能/成本数字来自 TypeSafe 自家 workflow eval 与 demo；长期定价可持续性文内承认需时间验证。
- 模型 **早期访问**，非权重开源；SDK 与 skills 为 MIT 开源。
