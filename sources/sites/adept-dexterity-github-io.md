# adept-dexterity.github.io（ADEPT 项目页）

- **标题：** ADEPT — Accelerating Dexterity via Pre-Training and Post-Training using Reinforcement Learning
- **类型：** site / project-page
- **URL：** <https://adept-dexterity.github.io/>
- **arXiv：** <https://arxiv.org/abs/2608.19182>
- **入库日期：** 2026-08-21
- **复核日期：** 2026-08-22
- **配套论文：** [ADEPT（arXiv:2608.19182）](../papers/adept_arxiv_2608_19182.md)

## 一句话摘要

NVIDIA / 密歇根大学提出的 **ADEPT** 官方站点：仿真 RL 在 16 种 primitive 上 pre-train 通用 reposing dexterity，再 structured post-train 成 FMB 插 peg / dish-rack 等下游专家，distill 成 RGB 或 visuo-tactile student **zero-shot** 部署到 Kuka–Allegro（23 DoF）与 Flexiv–Sharpa（29 DoF）；joint-space **Geometric Fabric** 贯穿 sim 与真机。

## 公开信息要点（截至 2026-08-22 复核）

- **机构：** NVIDIA Corporation；University of Michigan Robotics Department（Jayjun Lee、Ankur Handa 等）。
- **管线四阶段：** Pre-train → Post-train → Distill → Real world（交互式 method figure + rollout gallery）。
- **Pre-training：** 16 primitive shapes（见下表）；ADR + PBT；reach / grasp / lift / in-hand reorient / transport；~8B env steps。
- **Post-training：** BC distillation → critic warm-up → conservative PPO（actor LR 1e-3→1e-5 decay，clip 0.20→0.05 decay）；~3B env steps/task。
- **Distill：** 两阶段 vision curriculum；8-keypoint object-pose aux；Flexiv student 融合五指 TacMap + binary contact + SaTA-style FiLM。
- **Fabric：** full **Cspace** geometric fabric（非 DextrAH 式 5D PCA 子空间），policy 输出 per-joint relative delta；sim 与硬件同一低层控制器。
- **真机 episodic success（10 trials）：** Kuka–Allegro FMB star **5/10**、square/round **3/10**；Flexiv–Sharpa visuo-tactile square/round **8/10**（vision-only **3/10**）；dish placement **6/10**。
- **速度：** 5–10 s/trial vs FMB parallel-jaw pipeline 20–70 s（2–14×）。
- **代码 / 数据（步骤 2.5）：** 页头 **Code 按钮 → Coming soon**（`is-pending`，`aria-disabled="true"`）；**无** GitHub / Hugging Face / Zenodo URL。按 **宣称将开源 / 待发布** 处理。

### 16 种 pre-training primitives（项目页画廊）

| # | 形状 | 尺寸 |
|---|------|------|
| 01 | Cuboid | 50 × 100 × 100 mm |
| 02 | Cuboid | 50 × 50 × 100 mm |
| 03 | Cuboid | 25 × 100 × 100 mm |
| 04 | Cuboid | 25 × 50 × 100 mm |
| 05 | Cuboid | 25 × 25 × 100 mm |
| 06 | Cuboid | 10 × 100 × 100 mm |
| 07 | Sphere | 100 mm diameter |
| 08 | Sphere | 50 mm diameter |
| 09 | Capsule | 80 × 80 × 105 mm |
| 10 | Capsule | 80 × 80 × 90 mm |
| 11 | Capsule | 80 × 80 × 180 mm |
| 12 | Capsule | 50 × 50 × 150 mm |
| 13 | Capsule | 50 × 50 × 250 mm |
| 14 | Capsule | 20 × 20 × 220 mm |
| 15 | Cone | 100 × 100 × 100 mm |
| 16 | Cone | 50 × 50 × 100 mm |

规模跨度：50 mm 球～250 mm 杆（页面 toggle「True relative scale」可视化）。

### 真机 per-stage 累积成功率（`method-figure.js` Table 4 可视化）

各阶段为 **到达该阶段前所有阶段均成功** 的累积比例（10 trials）。

**Kuka–Allegro · FMB peg insertion**

| 阶段 | Star peg (%) | Square/Round (%) |
|------|-------------|------------------|
| Reaching | 100 | 100 |
| Grasping | 90 | 80 |
| Lifting | 80 | 60 |
| Reorienting | 80 | 40 |
| Aligning | 70 | 30 |
| Inserting (Overall) | **50 (5/10)** | **30 (3/10)** |

**Flexiv–Sharpa · FMB peg insertion**

| 阶段 | Visuo-tactile (%) | Vision-only (%) |
|------|-------------------|-----------------|
| Reaching | 100 | 100 |
| Grasping | 100 | 70 |
| Lifting | 100 | 50 |
| Reorienting | 90 | 30 |
| Aligning | 80 | 30 |
| Inserting (Overall) | **80 (8/10)** | **30 (3/10)** |

**Kuka–Allegro · dish-rack placement**

| 阶段 | Vision (%) |
|------|-----------|
| Reaching | 100 |
| Grasping (flip + regrasp) | 100 |
| Lifting | 80 |
| Reorienting | 70 |
| Aligning | 60 |
| Placing (Overall) | **60 (6/10)** |

### 站点非 PDF 证据

- **交互式 method figure：** 五 tab（Pre-train / Post-train / Distill / Real / Emergent dexterity）+ tooltip 解释 BC warm-up、clip/LR decay、8-keypoint aux、fabric Cspace 命令等。
- **Primitive 画廊：** 16 种形状 teacher reposing 视频 + 双 embodiment（Kuka–Allegro / Flexiv–Sharpa）切换。
- **Rollout gallery：** 按 stage（Pre-Training / Zero-Shot / Post-Training / Distillation / Real）与 task 过滤的 uncut rollout。
- **Qualitative analysis：** pre-trained vs from-scratch 抓型对比视频（contortion vs natural grasp）。
- **速度对比视频：** FMB pipeline 70 s@5× vs ADEPT 9 s 实时；20 s vs 5 s 并排。

## 为何值得保留

- **交互式 primitive 画廊：** 16 种 pre-training 形状与 teacher reposing 视频，比 PDF 附录更直观。
- **Post-training 伪代码：** 页面嵌入 `pre_train` / `post_train` 伪代码，便于与论文 Alg. 1 对照。
- **Per-stage 真机 breakdown：** SVG step plot 直接来自 Table 4，可定位失败阶段（如 vision-only 在 Reorienting 后骤降）。
- **触觉 decisive 证据：** tactile-combined 视频 + 8/10 vs 3/10 并列展示。

## 关联资料

- 论文归档：[`sources/papers/adept_arxiv_2608_19182.md`](../papers/adept_arxiv_2608_19182.md)
- Wiki 实体：[`wiki/entities/paper-adept-dexterity.md`](../../wiki/entities/paper-adept-dexterity.md)
- 任务域：[`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md)
- 手内重定向：[`wiki/methods/in-hand-reorientation.md`](../../wiki/methods/in-hand-reorientation.md)
