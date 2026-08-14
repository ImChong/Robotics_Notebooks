# HumanoidVLN 项目页（humanoid-vln.github.io）

> 来源归档（ingest 配套站点）

- **URL：** <https://humanoid-vln.github.io/>
- **标题：** HumanoidVLN — Physics-Grounded VLN Benchmark for Humanoid Robots
- **机构：** 越南人形机器人（VinMotion）；南加州大学（USC）
- **论文：** <https://arxiv.org/abs/2608.12860> — 归档见 [`sources/papers/humanoidvln_arxiv_2608_12860.md`](../papers/humanoidvln_arxiv_2608_12860.md)
- **入库日期：** 2026-08-14
- **一句话说明：** Isaac Sim 人形物理 VLN 落地页：四本体分层控制、≥100 m² 场景、MAA 指令、与 VLN-PE/VLNVerse 对照表、SR/NE/SPL/nDTW/OS/FR 协议。截至入库日 **无 Code / GitHub / Hugging Face 链接**。

## 开源核查（步骤 2.5，2026-08-14）

| 项 | 状态 |
|----|------|
| **论文承诺** | Abstract / 结论：*Code, benchmark, and data will be released upon acceptance* |
| **项目页 Code 区** | **无** GitHub、Hugging Face、Zenodo、ModelScope |
| **Footer / Resources** | 仅论文叙事、场景统计、评测协议；无下载 |
| **结论** | **宣称将开源 / 待发布**。勿写「已开源」；勿建 `sources/repos/`。放出后应补仓库归档与论文页时序图。 |

## 页面结构速记

1. **三缺口** — 传送不是 locomotion；多数场景塞不下双足；单次 VLM 指令会幻觉空间。
2. **四本体** — G1 / H1 / Internal-A / Internal-B 可互换加载同一环境；高层 PD/MPC，低层 RL 力矩。
3. **场景套件** — GRScenes 艺术家场景 + GS2Sim（gsplat）；n=87，全部 ≥100 m²。
4. **MAA** — Generator → Reviewer（scene-graph）→ Paraphraser（formal/natural/casual，保序）→ 人工终审。
5. **对照表** — 相对 R2R / VLN-CE / VLN-PE / VLNVerse：唯一同时标 Humanoid ✓、DoF 10–12、A+GS、MAA+Human、Hybrid 动作。
6. **协议** — SR / NE / SPL / nDTW / OS，并新增 **Fall Rate**。

## 关联资料

- 论文摘录：[`sources/papers/humanoidvln_arxiv_2608_12860.md`](../papers/humanoidvln_arxiv_2608_12860.md)
- Wiki 实体：[`wiki/entities/paper-humanoidvln.md`](../../wiki/entities/paper-humanoidvln.md)
- 仿真底座：[Isaac Sim](../../wiki/entities/isaac-sim.md)
- 被评模型入口：[NaVILA](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)
