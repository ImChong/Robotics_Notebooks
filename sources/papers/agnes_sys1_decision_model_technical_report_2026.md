# Agnes Sys1 1.0 Decision Model Technical Report

> 来源归档，对应 [独立论文详情](../../wiki/entities/paper-agnes-sys1-decision-model-technical-report.md)。

- **标题：** Agnes Sys1 1.0 Decision Model Technical Report
- **发布方：** AgnesAI-Labs；具体作者署名尚未核对（官方 README 表示 PDF 为 5 页）
- **论文 PDF：** https://github.com/AgnesAI-Labs/Agnes-Sys1-1.0-Technical-Report/blob/main/PAPER.pdf
- **官方可读摘要：** https://github.com/AgnesAI-Labs/Agnes-Sys1-1.0-Technical-Report/blob/main/REPORT.md
- **相关运行时：** https://github.com/AgnesAI-Labs/agnes-harness
- **核对范围：** 已通读 REPORT.md（全部 4 部分、公式与团队能力表），核对 AGH docs/guide/judgment.md；**无法直接读取 PDF**，未声称完整报告精读、权重代码复现。
- **通读：** 未完成（PDF 尚未核验；官方网页摘要已读完）
- **入库：** 2026-11-20

## 关键机制

1. **Qwen3.5-9B 基座：** 官方描述约 9B、32 层、hidden=4096；8 组“3 个 Gated DeltaNet + 1 个 full-attention”。
2. **块双向注意力：** 24 个 DeltaNet 维持 causal；8 个 full-attention 让共享 prefix P 内部双向，单个 question suffix S 可以读 P+S，但各 suffix 彼此隔离；前缀表示缓存可在多问题间共享。
3. **按请求候选概率：** 保留 LM head，只取候选标签 logits，并在本次候选上 softmax；概率未必校准，不等于 P(correct)。
4. **TTT 方案（非结果）：** 从材料构造可验证辅助多选题，在请求内仅更新接近末端的 RMSNorm 缩放参数；目标含 CE/Brier、跨视图 JS consistency、对基线的 KL anchoring、参数正则，失效/超时回退；更新后必须重建 KV cache 与 DeltaNet recurrent states，结束丢弃更新。
5. **评估计划：** 1/3/5 步训练比较、消融、ECE/Brier、abstention、P50/P95 延迟、显存与回退率等均为计划，**报告没有 TTT 实测增益**。

## 团队提供的能力指标（非独立复评）

| 指标 | Sys1 1.0 |
|---|---:|
| HLE-MultiChoice | 38.8% |
| When2Call | 95.2% |
| WildGuardTest | 93.6% |
| Omniscience accuracy (open-source subset) | 38.8% |
| Omniscience non-hallucination | 75.2% |
| AA Composite Index | 71.2 |

AA 由报告给定权重算得 71.15，四舍五入 71.2；相比表中 elephant-1 的 64.7 高 6.5 分，同时 non-hallucination **低 18.0 个百分点**。“Rank 1/2/3”不是当下实时排名。

## 开源边界与映射

- 报告仓库公开 PDF、README、REPORT.md；所核对资料未公开可验证的 Sys1 1.0 权重、训练脚本、独立推理/TTT benchmark 入口。
- [Sys1 论文节点](../../wiki/entities/paper-agnes-sys1-decision-model-technical-report.md)
- [AGH 主实体](../../wiki/entities/agnes-harness.md)；其 @agnes/judgment 为可选侧模型插件，不等于 Sys1 模型开源。
