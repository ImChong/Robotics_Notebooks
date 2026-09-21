# A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (Ross et al., 2011)

> 来源归档（ingest · AISTATS 2011 / PMLR）

- **标题：** A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning
- **类型：** paper
- **作者：** Stephane Ross, Geoffrey Gordon, Drew Bagnell
- **会议：** AISTATS 2011（Proceedings of Machine Learning Research, Vol. 15）
- **链接：** <https://proceedings.mlr.press/v15/ross11a.html>
- **PDF：** <http://proceedings.mlr.press/v15/ross11a/ross11a.pdf>
- **入库日期：** 2026-09-21
- **一句话说明：** 提出 **DAgger（Dataset Aggregation）**：把模仿学习与结构化预测归约到 **no-regret 在线学习**，训练 **平稳确定性策略**，在策略诱导分布上聚合专家标注，系统性缓解 BC 的 covariate shift 与 compounding error。

---

## 核心贡献（策展）

1. **问题设定：** 序列预测（模仿学习、结构化预测）中，未来观测依赖先前预测（动作），违反 i.i.d. 假设；纯 BC 在理论与实践中均表现差。
2. **算法：** 迭代式 **Dataset Aggregation（DAgger）**——用当前策略 rollout 收集访问状态，请专家回标，并入训练集重训；可视为在线 no-regret 算法。
3. **理论：** 在 reduction 假设下，no-regret 在线学习器必能在 **策略诱导的观测分布** 上找到低损失策略；相对 BC 将 horizon 误差从 $\mathcal{O}(\epsilon H^2)$ 量级改善到 $\mathcal{O}(\epsilon H)$（与后续文献常用表述一致）。
4. **实验：** 两个具挑战性的模仿学习任务 + 一个序列标注 benchmark，优于当时非平稳/随机策略的替代方案。

## 对 wiki 的映射

- [paper-ross-dagger](../../wiki/entities/paper-ross-dagger.md) — 论文实体页（本次新建）
- [DAgger（方法页）](../../wiki/methods/dagger.md) — 算法机制与工程落点
- [Behavior Cloning](../../wiki/methods/behavior-cloning.md) — covariate shift 对照
- [Imitation Learning](../../wiki/methods/imitation-learning.md) — IL 总览
- [imitation_learning.md](./imitation_learning.md) — 同主题 ingest 合集（条目 1 与本页互指）

## 参考来源（原始）

- PMLR  proceedings：<https://proceedings.mlr.press/v15/ross11a.html>
- PDF：<http://proceedings.mlr.press/v15/ross11a/ross11a.pdf>
