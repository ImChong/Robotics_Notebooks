## [2026-11-17] ingest | Karpo Decision 全尺寸结构化决策模型、源码路径与许可核对
- 归档 MachinePulse 的 0.8B–31B 六款公开模型及 `Decider` / System One 可运行推理入口，建立 [Karpo Decision](../wiki/entities/karpo-decision.md) 实体并交叉关联 AI Agent 评测。
- 核心机制：共享状态下对 `choice` / `noul` / `score` 候选一次性概率读出；Qwen3.5 和 Gemma 4 的编码/注意力路径需区分。
- 开放边界：推理仓 Apache-2.0，模型卡报告权重许可在 2026-11-12 后变更；性能是 Decision Index 口径，不推断真机控制结果。
