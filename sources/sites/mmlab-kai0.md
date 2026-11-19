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
- **最后核对：** 2026-11-19
- **核对范围：** 通读离线抓取的项目博客文本快照（`kai0_site.txt` 全文 132 行），核对 Mode Consistency 三分布叙事、Model Arithmetic / Stage Advantage / TDA 模块摘要、MSTD / SFR / Recover Cost / Throughput / Success Rate 图表说明与直播展示口径；未执行浏览器端 3D t-SNE 交互与视频播放。
- **通读：** 已完成
- **一句话说明：** χ₀ / kai0 的可视化项目页：Mode Consistency 三分布叙事、MA/SA/TDA 模块、成功率/恢复成本交互图，以及 100× 延时直播片段；链到 arXiv 与 OpenDriveLab/kai0。

## 开源状态（项目页核查 2026-08-05 / 2026-11-19）

- **代码：** Repository → `OpenDriveLab/kai0`（已公开）。
- **论文：** Report → arXiv:2602.09021。
- **数据 / 权重：** 博客正文写「We will release data, checkpoints, and host Challenge in 2026」；**以 GitHub README + HF/ModelScope 为准时，数据集与每任务 best ckpt 已在 2026-02 起陆续 Released**（见 [`sources/repos/kai0.md`](../repos/kai0.md)）。
- **Challenge：** 仍属预告，入库日未单独建赛页。

## 页面结构与核对要点

| 区块 | 内容要点 |
|------|----------|
| Hero / 直播 | 展平→折叠→挂衣三任务各 4 h 延时（100× 加速，关键段 2–5× 速） |
| Mode Consistency | $P_{\mathrm{train}}$ / $Q_{\mathrm{model}}$ / $P_{\mathrm{test}}$ 三分布互动三角 + 3D t-SNE；强调 DAgger 扩充 $P_{\mathrm{train}}$、推理平滑缩小 $Q_{\mathrm{model}}\to P_{\mathrm{test}}$ 失真 |
| Model Arithmetic | 权重空间插值并在 on-policy 数据上寻优混合权重，超越单子集最优与全量 oracle |
| Stage Advantage | 对比 Value-diff（独立值预测相减）、Direct 与 Direct+Stage（ours）；指标为 MSTD ↓、SFR ↑、SR ↑ |
| TDA 图表 | 展示 Heuristic DAgger vs 标准 DAgger（SR ↑ / Recover Cost ↓）、Spatio-temporal Augment（SR ↑ / TP ↑）与 Sync → In-chunk smooth → Temp smooth → RTC 递进 |
| Citation | 引用 `hkummlab2025kai0` 并链向 arXiv 与 GitHub |

## 对 wiki 的映射

- 主实体：[χ₀ / kai0（论文实体）](../../wiki/entities/paper-kai0.md)
- 论文摘录：[chi0_kai0_arxiv_2602_09021.md](../papers/chi0_kai0_arxiv_2602_09021.md)
- 仓库：[kai0.md](../repos/kai0.md)
