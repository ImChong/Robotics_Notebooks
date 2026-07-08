# 顶刊 T-RO 精选：2026上半年机器人操作学习的五项核心突破

> 来源归档（blog / 微信公众号）

- **标题：** 顶刊 T-RO 精选：2026上半年机器人操作学习的五项核心突破
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/nswA-jCGC3kr9iQjhRRuXQ
- **发表日期：** 2026-07-08
- **入库日期：** 2026-07-08
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；正文约 0.8 万字 / 17 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [wechat_shenlan_tro_manip_5_papers_2026-07-08.md](../raw/wechat_shenlan_tro_manip_5_papers_2026-07-08.md)
- **一句话说明：** 精选 T-RO 2026 上半年 **5 篇操作学习** 代表作，按 **数据规模化 → 三维/手物表征 → 无标签视频预训练 → 生成模型综述** 四条脉络串读；核心判断：规模化数据下 **表征与生成模型** 正重塑操作泛化，但数据多样性并非「越多越好」。

## 核心摘录（归纳，非全文）

### 问题重框

- 操作泛化瓶颈仍在 **数据怎么收、怎么表征、怎么从无标签视频迁移、生成模型如何进策略** 四条线并行演进。
- **读法：** 不按时间堆摘要，而按文内 **四条技术脉络** 组织；每篇对应独立 wiki 论文节点（见下表）。

### 四个分组（对应 5 篇）

| 组 | 篇数 | 核心问题 | 代表论文 |
|----|------|----------|----------|
| **01 数据规模化** | 1 | 任务/本体/演示者三维多样性如何影响 scaling？ | Is Diversity All You Need |
| **02 三维与手物表征** | 2 | 如何用 **SE(3) 等变** 与 **手物几何** 提升 OOD 泛化？ | Canonical Policy、DexRepNet++ |
| **03 无标签视频预训练** | 1 | 人类视频如何经 **图结构** 变成可迁移操作知识？ | G3M（GraphMimic 期刊版） |
| **04 生成模型综述** | 1 | 扩散/EBM/GAN 等在 LfD 中的选型与 OOD 设计？ | DGM Robot Learning Survey |

## 5 篇论文索引

### 01 — 数据规模化（1）

| # | 标题 | 机构 | 链接 |
|---|------|------|------|
| 01 | Is Diversity All You Need for Scalable Robotic Manipulation? | 港大；AgiBot；北航等 | arXiv:2507.06219 · IEEE T-RO 2026 |

### 02 — 三维与手物表征（2）

| # | 标题 | 机构 | 链接 |
|---|------|------|------|
| 02 | Canonical Policy: Learning Canonical 3D Representation for SE(3)-Equivariant Policy | 浙大；普渡 | arXiv:2505.18474 · <https://zhangzhiyuanzhang.github.io/cp-website/> |
| 03 | DexRepNet++: Learning Dexterous Robotic Manipulation With Geometric and Spatial Hand-Object Representations | 浙大；NUS 等 | arXiv:2602.21811 · <https://lqts.github.io/DexRepNet2/> |

### 03 — 无标签视频预训练（1）

| # | 标题 | 机构 | 链接 |
|---|------|------|------|
| 04 | Learning From Videos Through Graph-to-Graphs Generative Modeling for Robotic Manipulation（G3M） | 北京理工大学 | IEEE T-RO 2026 · CVPR 2025 GraphMimic |

### 04 — 生成模型综述（1）

| # | 标题 | 机构 | 链接 |
|---|------|------|------|
| 05 | A Survey on Deep Generative Models for Robot Learning From Multimodal Demonstrations | FAIR；英伟达等 | arXiv:2408.04380 · IEEE T-RO 2026 |

## 对 wiki 的映射

- [tro-manip-5-papers-technology-map](../../wiki/overview/tro-manip-5-papers-technology-map.md)（父节点 + Mermaid）
- [tro-manip-category-01-data-scaling](../../wiki/overview/tro-manip-category-01-data-scaling.md) … [tro-manip-category-04-generative-models-survey](../../wiki/overview/tro-manip-category-04-generative-models-survey.md)
- 论文实体：`wiki/entities/paper-tro-manip-01-diversity-scaling.md` … `paper-tro-manip-05-dgm-robot-learning-survey.md`
- **G3M（T-RO 2026）** 与 **GraphMimic（CVPR 2025）** 为同一研究线的会议/期刊版本，wiki 节点以 T-RO 标题 **G3M** 为主，勿与无关「Graph」论文合并。

## 可信度与使用边界

- 本文为 **微信公众号策展导读**，论文细节以 arXiv / IEEE Xplore / 项目页为准。
- 文内性能提升幅度须结合实验协议审慎解读。
- 原始抓取正文见 [wechat_shenlan_tro_manip_5_papers_2026-07-08.md](../raw/wechat_shenlan_tro_manip_5_papers_2026-07-08.md)。
