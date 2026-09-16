# WholeBodyWAM 项目页

> 来源归档（site）

- **标题：** WholeBodyWAM: Generalizing Pre-trained World-Action Priors to Humanoid Loco-Manipulation via WBC-Grounded Coordination
- **类型：** site
- **链接：** https://wholebodywam.github.io/
- **arXiv：** <https://arxiv.org/abs/2609.16644>
- **机构：** 香港中文大学（CUHK）；香港大学（HKU）；北京大学（PKU）；斐研究院（Phi Institute / Φ-Institute）
- **入库日期：** 2026-09-16
- **一句话说明：** 通过 WBC 语义接地与协调，将预训练世界—动作先验泛化到人形 loco-manipulation。
- **沉淀到 wiki：** [`wiki/entities/paper-wholebodywam.md`](../../wiki/entities/paper-wholebodywam.md)

## 开源状态

- **待发布**（步骤 2.5 核查，2026-09-16）。
- 项目页 **Paper / Code / Video** 区仅有 PDF 与演示视频；**无 GitHub、Hugging Face 或权重链接**。
- BibTeX 标注为匿名审稿引用，作者与发表细节待公开 release 后更新。

## 项目页要点

- **核心 slogan：** Preserve manipulation knowledge. Coordinate the whole body.
- **管线：** Preserve → Ground → Coordinate → Generalize。
- **架构：** 共享 DiT 从视觉历史与任务上下文生成未来视觉动力学、操作动作与 UWBC 命令；CASA 在可操作度下降时加强 manipulation→UWBC 注意力。
- **真机任务（8）：** CartServe、BoxTransfer、BasketCarry、DoorEntry、TowelPlace、TableCleanup、PlantWater、TeapotPour。
- **OOD 案例：** TowelPlace 支撑位移 10 cm 后二次抓取；TeapotPour 执行中切换目标杯子。
- **WBC 接口：** SONIC、AMO、GEAR（分别微调）。
- **对照基线：** Cosmos-3（仿真）、DreamZero（真机 OOD）。
