# 这是一篇把"RSI"讲明白的科普级综述！

> 来源归档（blog / 微信公众号 · Datawhale）

- **标题：** 这是一篇把"RSI"讲明白的科普级综述！
- **类型：** blog
- **作者：** 赵志民（Datawhale 成员；加拿大皇后大学博士）
- **原始链接：** https://mp.weixin.qq.com/s/rlfTKyWhALsNONhAwGih1A
- **发表日期：** 未稳定暴露（文内锚点 2026-07 GPT-5.6、2026-08 Motus2、2026 夏 AIDE² 等）
- **入库日期：** 2026-09-19
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **原始落盘：** [wechat_datawhale_rsi_survey_2026-09-19.md](../raw/wechat_datawhale_rsi_survey_2026-09-19.md)
- **一句话说明：** Datawhale 万字科普：用 **四层 RSI 标准**（持久改进 → 有界闭环 → 递归增益/ignition → 稳健可控）梳理 **五次边界推进**（记忆、权重、AI 打分、harness、训练/研究过程），并汇总 2026 夏 GPT-5.6 RSI Index、AIDE² Level 1 与 AI4AI-Bench 反向证据——强调 **闭环已出现、点火仍待证**。

## 核心摘录（归纳，非全文）

### 主判断

- **会自我改进 ≠ 会自我加速：** 固定题库上分数变高，或公司研发因 Agent 提速，都不等于改进能力本身复利。
- **OpenAI RSI Index（GPT-5.6 Sol 57.9%）** 是内部综合指标；完整题目、权重与聚合公式未公开，**不能**读成任务通过率或 RSI「完成度」。
- **当前共识（文内）：** 有界 RSI / 净正改进已有信号；**ignition**（改进后的系统更擅长改进）与 **开放式 RSI** 尚无充分公开证据。

### 四层标准（选型用）

| 层级 | 名称 | 可观察判据 | 文内 2026 状态 |
|------|------|------------|----------------|
| 1 | 持久改进 | 变化写入权重/记忆/harness/流程 | ✅ 多处出现 |
| 2 | 有界 RSI | 边界内 propose–eval–accept 多轮 | ✅ AIDE² 等自报 |
| 3 | 递归增益（ignition） | 新版本更会设计下一轮改进 | ⚠️ 无充分公开证据 |
| 4 | 稳健可控 | 隐藏评测泛化 + 对齐/审计 | ❌ 远未解决 |

### 五次边界推进（叙事框架，时间重叠）

| 次 | 时期 | 代表 | 改动对象 | 边界 |
|----|------|------|----------|------|
| 0 | 1981–2017 | EURISKO、AlphaZero | 规则/权重 | 评估函数与人类写死沙盒 |
| 1 | 2022–23 | Reflexion | 外部情景记忆 | 参数不变；无新版本 |
| 2 | 2022–24 | STaR、SPIN | 模型权重 | 目标与「红笔」仍人类 |
| 3 | 2022–25 | CAI、Self-Rewarding、Meta-Rewarding | 反馈信号 | 自评漂移（EURISKO 2024 版） |
| 4 | 2023–26 | OPRO、ADAS、DGM、Self-Harness、AgentX | harness / 工作流 | 验证器与业务目标仍外部 |
| 5 | 2025–26 | SEAL、WebEvolver、Motus2、OpenAI/Google 研发 Agent | 训练材料、环境、研究执行 | 研究方向与采纳权仍人类 |

### 2026 夏关键信号（宜打折读）

| 工作 | 文内主张 | 证据边界 |
|------|----------|----------|
| GPT-5.6 RSI Index | 57.9% vs 41.7% | 内部指标；细节未公开 |
| Bilevel Autoresearch | 外层改搜索机制；验证损失降幅 ~5× | 规模受控实验 |
| RHI | 低预算 Agent 超未优化高预算；推理成本 −60% | 30 合成 ML 研究任务 |
| AIDE²（Weco） | 100 步改内层 harness；称 **Level 1** 有界 RSI | **团队博客自报**；未 peer review；**未过 ignition**；复杂度膨胀 |
| AI4AI-Bench | 最强 0.250/1.0；多数未改核心学习算法 | 反向证据 |
| Motus2 | 真机成功率 65%→75%；权重级有界闭环 | 世界模型/价值冻结；非开放式 RSI |

### 四道门（为何 ignition 难）

1. **验证器锚** — 评估器必须在被改进系统控制范围外（隐藏测试、编译器、证明器）。
2. **分布外泛化** — 自生成数据 → 模型坍缩；固定题库 → harness 过拟合。
3. **递归增益** — 分数 ↑ ≠ 更会设计训练/改进本身（AIDE² 未证；AI4AI-Bench 浅）。
4. **能力–控制同步** — 权限越高，审计/回滚/对齐须同步；AutoResearchEval 缺稳定元认知循环。

## 对 wiki 的映射

- [rsi-four-tier-five-pushes](../../wiki/queries/rsi-four-tier-five-pushes.md)（本次升格主页面）
- [recursive-self-improvement](../../wiki/concepts/recursive-self-improvement.md) — 宏观 RSI 与 Anthropic 治理视角互补
- [awesome-rsi](../../wiki/entities/awesome-rsi.md) — artifact × mode 方法索引
- [paper-motus2](../../wiki/entities/paper-motus2.md) — 第五次推进中的具身有界闭环实例
- [ai-auto-research](../../wiki/concepts/ai-auto-research.md)、[real-robot-policy-autoresearch-harness](../../wiki/queries/real-robot-policy-autoresearch-harness.md)

## 可信度与使用边界

- 科普综述 + 2026 夏 arXiv/博客混合来源；**AIDE²、OpenAI RSI Index 等宜作阶段性信号，非已沉淀共识**。
- 文内 mermaid/表格为策展压缩；工程细节以原文与论文为准。
- 无项目页需步骤 2.5 核查；所引仓库开源状态以各 `sources/repos/` 与实体页为准。

## 推荐继续阅读（外部）

- OpenAI GPT-5.6 与 RSI Index：<https://openai.com/index/gpt-5-6/>
- Weco AIDE² Level 1 报告：<https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement>
- Lilian Weng Harness Engineering：<https://lilianweng.github.io/posts/2026-07-04-harness/>
- Reflexion：<https://arxiv.org/abs/2303.11366> · STaR：<https://arxiv.org/abs/2203.14465> · DGM：<https://arxiv.org/abs/2505.22954>
- AI4AI-Bench：<https://arxiv.org/abs/2608.20318> · AutoResearchEval：<https://arxiv.org/abs/2608.14905>
