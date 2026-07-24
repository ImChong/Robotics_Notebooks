# EgoVerse（项目页 · egoverse.ai）

> 来源归档（ingest）

- **标题：** EgoVerse — Human data from around the world, built for robot learning
- **类型：** site / dataset ecosystem / consortium
- **官方入口：** <https://egoverse.ai/>
- **论文：** <https://arxiv.org/abs/2604.07607>
- **代码：** <https://github.com/GaTech-RL2/EgoVerse>（MIT；数据处理 / 训练 / 评测）
- **数据浏览器：** <https://partners.mecka.ai/egoverse>
- **机构 / 联盟：** Georgia Tech RL²、Stanford REAL、UC San Diego Wang Lab、ETH Zürich CVG & SRL；产业伙伴含 Meta Reality Labs Research、Mecka AI、Scale AI 等（以项目页 Consortium / Team 为准）
- **入库日期：** 2026-07-24
- **一句话说明：** 面向机器人学习的 **egocentric 人类示教** 生态站点：展示活数据集规模、旗舰任务与跨实验室人→机迁移验证入口，并链到 arXiv 论文与官方 GitHub。

## 开源与数据开放核查（步骤 2.5 · 入库日 2026-07-24）

| 项 | 状态 |
|----|------|
| **项目页 Code** | **已开源** — 显式链到 [GaTech-RL2/EgoVerse](https://github.com/GaTech-RL2/EgoVerse) |
| **论文** | **已公开** — arXiv:2604.07607 |
| **数据集访问** | **部分开放 / 受控** — Dataset browser（partners.mecka.ai）+ 仓库内 SQL 过滤 + Cloudflare R2 / S3 同步；需按官方 README 配置云凭证，**非**无门槛匿名全量镜像 |
| **评测验证** | 项目页展示多实验室真机 rollout（object-in-container / cup-on-saucer / bag-grocery 等） |

## 页面摘录要点

- **定位：** Capture · Dataset · Evaluation；「living」数据集，由联盟持续扩展，服务 human-to-robot transfer 的可复现研究。
- **规模快照（页面 Dataset Snapshot）：** **1,362 h** 人示教、**~80k** episodes、**1,965** tasks、**240** scenes、**2,087** unique demonstrators。
- **标注卖点：** 准确相机位姿、3D 头部跟踪、稠密语言标注；旗舰标准化任务 + 开放场景任务并存。
- **跨具身验证：** Georgia Tech RL²、Stanford REAL、UCSD Wang Lab、ETH CVG & SRL 等在不同任务/形态上复现协议。

## 对 wiki 的映射

- [EgoVerse 论文实体](../../wiki/entities/paper-egoverse.md)
- [EgoWAM](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md) — 野外人数据来源
- [HumanNet Table 1 语料对照](../../wiki/comparisons/humannet-table1-human-video-corpora.md)
- [Imitation Learning](../../wiki/methods/imitation-learning.md)
- [Manipulation](../../wiki/tasks/manipulation.md)

## 交叉链接（sources 互指）

- 论文归档：[egoverse_arxiv_2604_07607.md](../papers/egoverse_arxiv_2604_07607.md)
- 仓库归档：[egoverse.md](../repos/egoverse.md)
