# χ₀ / kai0 — HKU MMLab 项目博客

> 来源归档（ingest）

- **标题：** $\chi_{0}$: A Live-Stream Robotic Teamwork for Clothing Manipulation from Zero to Hero
- **类型：** site / project blog
- **URL：** <https://mmlab.hk/research/kai0>
- **论文：** <https://arxiv.org/abs/2602.09021>
- **代码：** <https://github.com/OpenDriveLab/kai0>
- **机构叙事：** HKU MMLab Community；PDF 署名 Kinetix AI；仓属 OpenDriveLab
- **发布标注：** 博客文首 Dated December 24, 2025（与 arXiv 2026-02 并行的工程叙事页）
- **入库日期：** 2026-08-05
- **一句话说明：** χ₀ / kai0 的可视化项目页：Mode Consistency 三分布叙事、MA/SA/TDA 模块、成功率/恢复成本交互图，以及 100× 延时直播片段；链到 arXiv 与 OpenDriveLab/kai0。

## 开源状态（项目页核查 2026-08-05）

- **代码：** Repository → `OpenDriveLab/kai0`（可点开）。
- **论文：** Report → arXiv:2602.09021。
- **数据 / 权重：** 博客正文写「We will release data, checkpoints, and host Challenge in 2026」；**以 GitHub README + HF/ModelScope 为准时，数据集与每任务 best ckpt 已在 2026-02 起陆续 Released**（见 [`sources/repos/kai0.md`](../repos/kai0.md)）。
- **Challenge：** 仍属预告，入库日未单独建赛页。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Hero / 直播 | 展平→折叠→挂衣三任务 4 h 延时；关键片段 2–5× |
| Mode Consistency | $P_{\mathrm{train}}$ / $Q_{\mathrm{model}}$ / $P_{\mathrm{test}}$ 三角；3D t-SNE |
| Methodology | Model Arithmetic、Stage Advantage、Train-Deploy Alignment |
| Charts | Success Rate / Recover Cost / Throughput 等交互图 |
| Citation | 指向 arXiv 与 GitHub |

## 对 wiki 的映射

- 主实体：[χ₀ / kai0（论文实体）](../../wiki/entities/paper-kai0.md)
- 论文摘录：[chi0_kai0_arxiv_2602_09021.md](../papers/chi0_kai0_arxiv_2602_09021.md)
- 仓库：[kai0.md](../repos/kai0.md)
