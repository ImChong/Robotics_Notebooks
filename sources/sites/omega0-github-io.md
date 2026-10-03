# ω-0 / OMEGA-0 Project Page

> 来源归档（ingest；2026-10-03 复核开源链接）

- **标题：** ω-0: A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation
- **类型：** project site
- **官方入口：** <https://gentlefress.github.io/OMEGA-0_page/>
- **论文：** <https://arxiv.org/abs/2608.06375>（PDF：<https://arxiv.org/pdf/2608.06375>）
- **代码：** <https://github.com/gentlefress/Omega-0>（[归档](../repos/omega-0.md)）
- **数据集：** <https://huggingface.co/datasets/keycharon/omega-HOME>（[归档](../datasets/omega-home.md)）
- **机构：** NTU MARS Lab；北京大学；北京智源人工智能研究院（BAAI）；香港科技大学广州校区
- **核查日期：** 2026-10-03
- **一句话说明：** 官方方法与结果页，并链接至已公开的训练/部署代码和 ω-HOME 数据集。

## 开源状态（2026-10-03）

| 项 | 状态 |
|----|------|
| 论文 | arXiv 2608.06375 |
| 代码 | 项目主页 GitHub 链接现指向公开仓库；含训练、推理、采集与 G1 部署代码 |
| 数据集 | 项目主页链接至 Hugging Face 的 ω-HOME 页面 |
| 预训练权重 | 官方仓库 README 的 TODO 仍标注待发布 |
| 许可证 | 代码仓标 MIT；HF 数据集页标 MIT；第三方组件仍依各自许可证 |

此前 2026-08-10 的核查记录为 Code / Dataset WIP；这是当时页面状态。到 2026-10-03，项目页已链接上述代码仓和数据集页面。

## 页面内容

| 区块 | 内容要点 |
|------|----------|
| Abstract | 潜空间 foresight + 扩散全身动作；SONIC 兼容动作 latent |
| Method | Action-token VLM 预训练 → 联合 world–action 学习 → 真机微调与 RTC |
| Results | 11 项家务评测；项目页报告成功率 81.8%、任务进度 90.3% |
| Dataset | ω-HOME：项目页报告 40.3 小时、24 任务、4,827 episodes、六类同步模态 |
| Demo | 真机家务任务视频 |
| Citation | BibTeX（`li2026omega0`） |

## 对 wiki 的映射

- 论文：[`sources/papers/omega0_arxiv_2608_06375.md`](../papers/omega0_arxiv_2608_06375.md)
- 代码：[官方仓库归档](../repos/omega-0.md)
- 数据集：[ω-HOME 归档](../datasets/omega-home.md)
- 论文节点：[`wiki/entities/paper-omega-0.md`](../../wiki/entities/paper-omega-0.md)
