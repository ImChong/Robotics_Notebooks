# D-JEPA 官方仓库

> 来源归档（repo）

- **标题：** D-JEPA: A Decision-Aligned Latent World Model
- **类型：** repo
- **链接：** https://github.com/NEBULIS-Lab/D-JEPA
- **arXiv：** <https://arxiv.org/abs/2609.24749>
- **入库日期：** 2026-10-18
- **一句话说明：** 用实际执行结果学习候选未来之间的关系，修正潜在距离与任务成败不一致的问题，让世界模型选对能成功的动作。
- **沉淀到 wiki：** [`wiki/entities/paper-d-jepa-decision-aligned-latent-world-model.md`](../../wiki/entities/paper-d-jepa-decision-aligned-latent-world-model.md)

## 开源状态

- **已开源**：仓库可达（git ls-remote），项目页 HTTP 200。 仓库内容以 README 与 release 为准（本次仅用 `git ls-remote` 核实可达，未逐个阅读 README）。

## 固定版本源码复核（2026-11-13）

- 版本：[9b59aaa6d908271ca9b66dd3d98ae558404e7f9f](https://github.com/NEBULIS-Lab/D-JEPA/tree/9b59aaa6d908271ca9b66dd3d98ae558404e7f9f)。已读根README、cli/evaluate.py、evaluation/inference.py、models/relational.py、models/ordinal.py全文，以及exact_realization.py完整forward_from_futures。
- CLI先predict(checkpoint,inputs)完成选择才打开labels；load_profile严格载tensor权重/config；候选features/base_scores/IDs→DecisionAligner→gated_positions→selected_ids→可选标签统计。cached replay明确不执行新仿真。
- RelationalAligner接63×192双骨干terminal差与rank，set Transformer无候选位置编码、低秩tanh修正±.2。ExactRealizationWorldModel按ID打破tie，gate优势四位小数，terminal radius=(rank+1)/64，早期future不动，并断言native rank/argmin一致。
- 公开task profiles、decision supervision ZIP、训练与native执行工具；2B/robot等上游predictor和设备依赖须另准备。本次无权重下载/GPU复现，不能从源码可达推出所有实机可直接运行。
