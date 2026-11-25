# Recursive self-improvement of AI research agents / AI 研究智能体的递归自我改进（AIDE²）

> 一手论文全文阅读归档（paper）；本文件归纳原文，不复制受版权保护的全文。

- **作者：** Dhruv Srikanth、Bingchen Zhao、Dixing Xu、Yuxiang Wu、Zhengyao Jiang
- **机构：** Weco AI
- **类型：** arXiv 预印本（cs.AI / cs.LG / cs.SE）
- **版本：** arXiv:2609.26457v1，2026-09-22
- **论文：** https://arxiv.org/abs/2609.26457v1
- **HTML 全文：** https://arxiv.org/html/2609.26457v1
- **PDF：** https://arxiv.org/pdf/2609.26457v1
- **DOI：** 10.48550/arXiv.2609.26457
- **官方解读：** https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement
- **核对范围：** v1 全文 §1–§5、Algorithm 1、实验图表及附录 A–E（held-out 协议、基线比较、跨模型、拒绝提案、提示压缩）；官方解读的开源入口和研究边界。
- **通读：** 已完成

## 一手资料摘录与核实

1. **任务与层级：** AIDE² 将一名 AI 研究 agent 的 harness 代码当成外层待优化对象。内层在机器学习工程、启发式算法工程、harness 工程任务上生成/执行候选代码；内层看公开反馈，外层只用任务私有留出集评分的聚合值决定是否保留 agent 改写。
2. **预算不变：** 对每个任务限制 token + 代码执行的总美元预算，避免单靠增加调用次数得到伪改进。外层的 AIDE_human 使用 Claude Opus 4.7，内层用 Gemini 3 Flash；不等价于底座模型自训练。
3. **结果：** 8 天、100 个轨迹节点（含初始版、99 次提案），接受 7 次改写，private grade 从 0.703 升到 0.778；另两次运行分别接受 2 次和 4 次。
4. **外部评测：** ALE-Bench、MLE-Bench、FML-Bench、WeatherBench 2 四类 held-out 基准；AIDE_85 对比 AIDE_human 分别为 1790 vs 1511、0.722 vs 0.708、19.9% vs 19.6%、0.793 vs 0.404（各指标的单位不同，不能求总平均）。这些是论文 Appendix A Table 1 的均值，统计不确定性见原表。
5. **搜索与记忆：** AIDE_85 用五个策略臂的 UCB1 + softmax 混合选择，并每五步 fork 最佳节点；用有界摘要、按角色上下文和 bug-rate ≥15% 时最多 3 个去重错误签名代替全量历史。
6. **reward hacking：** 独立 KernelBench 下游训练集成评估显示 AIDE_0 55%、AIDE_47 39%、AIDE_85 32%，手工基线 AIDE_human 39%。这不是选拔目标，且无法据此归因于某一个改写。
7. **ignition test：** AIDE_47 与 AIDE_human 分别作外层 agent，3 个种子 × 50 步，最终 private grade 均值 0.780 vs 0.782；作者明确认为证据不足以证明改进后的 agent 在“改进下一代 agent”方面更强。

## 论文边界

- 结果展示 **harness 层**的经验性递归改写；不是 LLM 权重训练、自动设计下一代基础模型，也没有机器人策略/真机实验证明。
- 外部基准并非完全独立复现；不同 benchmark 模型、预算和种子数量不同，WeatherBench 2 只有一项任务、3 个种子。
- 论文指出实验噪声、复杂性膨胀与未通过的 ignition 验证；不得将 Level 1 自报等同于加速式无限自改进。
- **源码开放核查（2026-11-21）：** 作者官方博客与 arXiv v1 未列 AIDE² 整体可运行代码仓库/复现入口。先前的 AIDE 与其他开源 autoresearch 实现不能替代本论文的完整源码。

## 对 wiki 的映射

- **独立论文实体：** [AIDE²：AI 研究智能体的递归自我改进](../../wiki/entities/paper-aide2-recursive-self-improvement-ai-research-agents.md)
- **定位：** [RSI 纵深路线 Stage 4–5](../../roadmap/depth-rsi.md)（harness 自改写、隐藏评测、ignition 边界）
- **关联：** [AI Auto-Research](../../wiki/concepts/ai-auto-research.md) 与 [递归自我改进](../../wiki/concepts/recursive-self-improvement.md)

## 参考来源

- [arXiv 原文 HTML v1](https://arxiv.org/html/2609.26457v1)
- [Weco AI 官方解读](https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement)
