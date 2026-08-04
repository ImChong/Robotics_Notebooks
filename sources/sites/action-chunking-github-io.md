# action-chunking.github.io（Why Action Chunking Improves BC 项目页）

- **标题：** Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?
- **类型：** site / project-page
- **URL：** <https://action-chunking.github.io/>
- **PDF（项目页托管）：** <https://action-chunking.github.io/static/action_chunking.pdf>
- **演示幻灯：** <https://action-chunking.github.io/static/presentation.pdf>
- **配套论文归档：** [`sources/papers/why_action_chunking_improves_bc_corl2026.md`](../papers/why_action_chunking_improves_bc_corl2026.md)
- **机构：** Politecnico di Milano（¹）、UC Berkeley（²）
- **作者：** Filippo Lazzati¹、Kyle Stachowicz²、William Chen²、Alberto Maria Metelli¹、Andrew Wagenmaker²、Sergey Levine²
- **入库日期：** 2026-08-04

## 一句话摘要

CoRL 2026 论文项目页：用仿真（LIBERO / Robomimic）与 Franka 真机消融，论证 action chunking 的收益主要来自 **延迟观测条件化（delayed policy）** 与 **隐式集成（implicit ensembling）**，而非常见的「时序一致性 / 有效地平线缩短 / 表征学习」叙事；并提出 **Randomized Delay Ensemble（RDE）** 与显式延迟策略集成，可在多数设定匹配甚至超过标准 chunk 执行。

## 开源状态（步骤 2.5，截至 2026-08-04）

| 资源 | 状态 |
|------|------|
| 项目页 PDF / presentation | **已发布**（`./static/action_chunking.pdf`、`./static/presentation.pdf`） |
| arXiv | **Coming soon**（页上按钮未挂编号） |
| Code | **Coming soon**（页上 GitHub 按钮无可用 URL） |

**结论：宣称将开源 / 代码与 arXiv 待发布。** 复现入口以项目页 PDF 与叙述为准；wiki「源码运行时序图」标 **不适用**，待正式 release 后补。

## 公开信息要点

- **会议元数据：** PDF Producer 标注 *Proceedings of the 10th Conference on Robot Learning (CoRL 2026)*。
- **三条机制叙事（页首 Mechanism 1–3）：**
  1. 人类演示的非马尔可夫性可用 \(a_t \mid o_{t-n}\) 捕捉；**不必**要求 chunk 内联合时序一致性。
  2. 复合误差上界从 \(\Omega(2^H\epsilon)\) 改善到 \(\mathcal{O}((k+1)^{H/k}\epsilon)\)，但同样被 delayed policy 捕获（并非「只每 \(k\) 步决策」的有效地平线故事）。
  3. chunk 训练同时拟合多种时延关系 → **隐式集成**；部署时用 RDE（每步随机选延迟索引）可在不执行整段 chunk 的情况下复现收益。
- **真机：** Franka Emika，三任务（carrot in bowl / bread toaster / sushi in cup），各 50 demos、50 rollouts；对比 single-step / delayed / action chunking / RDE。
- **BibTeX：** `@article{lazzati2026chunking, ... year={2026}}`（无 arXiv id）。

## 为何值得保留

- 直接冲击本库 [Action Chunking](../../wiki/methods/action-chunking.md) 方法页里「平滑 / 降复合误差 / 延迟缓冲」的常见工程读法——给出可操作的 **延迟策略与 RDE 替代部署**。
- 与 ACT / Diffusion Policy / VLA 默认 chunk 训练形成「机制级」对照，而非又一篇工程管线。

## 关联资料

- 论文归档：[`sources/papers/why_action_chunking_improves_bc_corl2026.md`](../papers/why_action_chunking_improves_bc_corl2026.md)
- 沉淀实体：[`wiki/entities/paper-why-action-chunking-improves-bc.md`](../../wiki/entities/paper-why-action-chunking-improves-bc.md)
