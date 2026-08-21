# adept-dexterity.github.io（ADEPT 项目页）

- **标题：** ADEPT — Accelerating Dexterity via Pre-Training and Post-Training using Reinforcement Learning
- **类型：** site / project-page
- **URL：** <https://adept-dexterity.github.io/>
- **arXiv：** <https://arxiv.org/abs/2608.19182>
- **入库日期：** 2026-08-21
- **配套论文：** [ADEPT（arXiv:2608.19182）](../papers/adept_arxiv_2608_19182.md)

## 一句话摘要

NVIDIA / 密歇根大学提出的 **ADEPT** 官方站点：仿真 RL 在 16 种 primitive 上 pre-train 通用 reposing dexterity，再 structured post-train 成 FMB 插 peg /  dish-rack 等下游专家，distill 成 RGB 或 visuo-tactile student **zero-shot** 部署到 Kuka–Allegro（23 DoF）与 Flexiv–Sharpa（29 DoF）；joint-space **Geometric Fabric** 贯穿 sim 与真机。

## 公开信息要点（截至 2026-08-21 核查）

- **机构：** NVIDIA Corporation；University of Michigan Robotics Department（Jayjun Lee、Ankur Handa 等）。
- **管线四阶段：** Pre-train → Post-train → Distill → Real world。
- **Pre-training：** 16 primitive shapes（50 mm 球～250 mm 杆）；ADR + PBT；reach / grasp / lift / in-hand reorient / transport。
- **Post-training：** BC distillation → critic warm-up → conservative PPO（actor LR 1e-5，clip 0.05）；相对 naive fine-tune 避免 collapse。
- **Fabric：** full **Cspace** geometric fabric（非 DextrAH 式 5D PCA 子空间），sim 与硬件同一低层控制器。
- **真机结果（10 trials）：** Kuka–Allegro FMB star **5/10**、square/round **3/10**；Flexiv–Sharpa visuo-tactile square/round **8/10**（vision-only **3/10**）；dish placement **6/10**。
- **速度：** 5–10 s/trial vs FMB parallel-jaw pipeline 20–70 s（2–14×）。
- **代码 / 数据（步骤 2.5）：** 页头 **Code 按钮 → Coming soon**；**无** GitHub / Hugging Face URL。按 **宣称将开源 / 待发布** 处理。

## 为何值得保留

- **交互式 primitive 画廊：** 16 种 pre-training 形状与 teacher reposing 视频，比 PDF 附录更直观。
- **Post-training 伪代码：** 页面嵌入 pre_train / post_train 伪代码，便于与论文 Alg. 1 对照。
- **Per-stage 真机 breakdown：** Reach→Grasp→Lift→Reorient→Align→Insert 累积成功率表。

## 关联资料

- 论文归档：[`sources/papers/adept_arxiv_2608_19182.md`](../papers/adept_arxiv_2608_19182.md)
- 任务域：[`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md)
- 灵巧操作：[`wiki/methods/dexterous-manipulation-rl.md`](../../wiki/methods/dexterous-manipulation-rl.md)（若存在）
