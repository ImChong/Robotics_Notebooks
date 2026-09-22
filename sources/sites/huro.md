# HuRo Project Page

- **标题：** HuRo: Robotizing Human Videos for Scalable VLA Pretraining
- **类型：** site
- **链接：** <https://3587jjh.github.io/HuRo/>
- **论文：** [arXiv:2609.10706](https://arxiv.org/abs/2609.10706)（CoRL 2026）
- **机构：** RLWRLD；延世大学（Yonsei University）
- **代码：** <https://github.com/3587jjh/HuRo>
- **数据集：** Hugging Face **Coming soon**（页内与 README 一致）
- **入库日期：** 2026-09-11（索引）；**再核日期：** 2026-09-22
- **一句话说明：** 三阶段机器人化流水线把五源 egocentric 人视频转成 ALLEX 对齐观测与重定向动作；630K episode / 142M 帧；VLA 预训练后真机四项任务 Overall **51.5→80.3%**、OOD **34.9→72.2%**。
- **沉淀到 wiki：** [`wiki/entities/paper-huro.md`](../../wiki/entities/paper-huro.md)
- **交叉归档：** [`sources/repos/huro.md`](../repos/huro.md)、[`sources/papers/huro_arxiv_2609_10706.md`](../papers/huro_arxiv_2609_10706.md)

## 开源核查（2026-09-22）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（CoRL 2026） |
| GitHub `3587jjh/HuRo` | **已开源**（Apache-2.0）：10 阶段机器人化流水线 → LeRobot V2.0 |
| HuRo 预构建数据集 | **待发布**（README badge：Data coming soon） |
| VLA 预训练 / 微调权重 | **未见** 公开 checkpoint |
| 商用 | **不可**（依赖许可限制，见仓库 `THIRD_PARTY_NOTICES.md`） |

**结论：部分开源** — 流水线代码可跑；大规模 HuRo 语料与 VLA 权重仍待发布。

## 页内核心数字（2026-09-22 抓取）

| 指标 | 数值 |
|------|------|
| Robotized episodes | 630K |
| Processed frames | 142M（≈1,317 h @ 30 fps） |
| 人视频来源 | 5（EgoDex 55%、EgoVerse 27%、Ego4D 10%、Ego10K 6%、EPIC-Kitchens 2%） |
| Overall completion（scale PT） | 51.5% → 80.3% |
| ID completion | 68.1% → 88.4% |
| OOD completion | **34.9% → 72.2%** |
| 真机评测 | ALLEX 四项 + Diverse P&amp;P；对比 π₀.₅、GR00T N1.6 |
