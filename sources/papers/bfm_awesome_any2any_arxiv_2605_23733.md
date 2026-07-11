# Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking

> 来源归档（ingest · awesome-bfm-papers 第 42/42）

- **标题：** Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking
- **类型：** paper
- **BFM 分类：** 04 Adaptation · Fine-tuning Techniques（[awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)）
- **出处：** 2026 · arXiv
- **论文链接：** <https://arxiv.org/abs/2605.23733>
- **代码/项目：** N/A（LimX Dynamics 组织）
- **索引来源：** [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)（2026-07-11 列表刷新新增）
- **入库日期：** 2026-07-11
- **一句话说明：** 把单源 WBT 专家经运动学对齐 + LoRA 动力学适配迁到新机体，约 1% 全量训练成本——BFM 适应线中 **跨具身 PEFT** 代表。

## 核心摘录（策展，非全文）

- **在 awesome 列表中的位置：** 04 Adaptation → Fine-tuning Techniques，编号 **42/42**（2026-07 较初版 41 篇新增）。
- **方法要点：** 冻结源 WBT 骨干；运动学层对齐髋轴/闭链/观测顺序；动力学敏感模块上低秩适配（LoRA/PEFT）。
- **与 BFM taxonomy 关系：** 属于 **适应·微调** 线——假设已有 goal-conditioned 或 scaling tracker 预训练 checkpoint，用 **后训练** 而非从头预训练解决新机部署。
- **本库深读：** 方法细节、真机结果与 SONIC 对照见独立 ingest [`any2any_arxiv_2605_23733.md`](any2any_arxiv_2605_23733.md)。

## 对 wiki 的映射

- [paper-any2any-cross-embodiment-wbt](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md) — 完整实体页（复用，不另建 paper-bfm-42）
- [behavior-foundation-model](../../wiki/concepts/behavior-foundation-model.md) — 适应·微调线
- [bfm-category-04-adaptation](../../wiki/overview/bfm-category-04-adaptation.md) — 分类 hub
- [cross-embodiment-transfer-strategy](../../wiki/queries/cross-embodiment-transfer-strategy.md) — 三路径选型

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2605.23733>
- 策展列表：<https://github.com/friedrichyuan/awesome-bfm-papers>
- 本库深读 source：[any2any_arxiv_2605_23733.md](any2any_arxiv_2605_23733.md)
