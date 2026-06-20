# 盘点 | VLN 领域最具影响力的 10 项代表性研究

> 来源归档（blog / 微信公众号）

- **标题：** 盘点 | VLN 领域最具影响力的 10 项代表性研究
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/2_dYaN6IeWn_vvS_jmGqRQ
- **发表日期：** 2026-06-20
- **入库日期：** 2026-06-20
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）；正文约 0.89 万字 / 19 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **关联姊妹篇：** [VLN 四范式新手复现](wechat_shenlan_vln_repro_four_paradigms_2026.md)、[10 年 VLN 三大技术拐点](https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247486817&idx=1&sn=39e5ea88bb60739f01c6ae02bbedafed)（文内推荐区外链，未单独 ingest）
- **一句话说明：** 按 **两组入口**（01 数据集与仿真平台、02 算法框架）串读 VLN 领域 2018–2024 最具代表性的 10 项工作；核心判断：从 R2R 离散导航图到 NaVid 单目 RGB 视频流端到端决策，贯穿 **逐步剥离显式辅助信号（导航图、深度、里程计）** 的「减负」演进线。

## 核心摘录（归纳，非全文）

### 问题重框

- **VLN 减负史：** 2018 R2R 确立可量化基准后，任务从离散图跳转 → 连续底层动作 → 高层目标定位，模型从 Seq2Seq → 预训练 Transformer → 拓扑建图 → 大规模数据生成 → VLM 视频流端到端。
- **读法：** 不按时间堆摘要，而按 **两组入口** 组织：基础设施（测什么、在哪测、动作空间如何定义）与算法框架（对齐、记忆、规划、扩数据、接 VLM）。
- **收束判断：** 三阶段演进——(1) 任务定义与基础设施；(2) 预训练范式与架构创新；(3) 数据扩展与大模型融合。下一阶段核心挑战：VLM 通用能力与导航时序决策的结合，以及降低仿真依赖、提升真机泛化。

### 两个分组（对应 10 篇）

| 组 | 篇数 | 核心问题 | 代表论文 |
|----|------|----------|----------|
| **01 数据集与仿真平台** | 3 | **测什么、在哪测、动作空间如何定义？** | R2R、VLN-CE、REVERIE |
| **02 算法框架** | 7 | **模型如何对齐、记忆、规划、扩数据、接 VLM？** | PREVALENT → NaVid |

## 10 篇论文索引

### 01 — 数据集与仿真平台（3）

| # | 标题 | 机构 | 出处 | 链接 |
|---|------|------|------|------|
| 01 | Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments | 澳大利亚国立大学、阿德莱德大学等 | CVPR 2018 | arXiv:1711.07280 |
| 02 | Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments | 俄勒冈州立大学、佐治亚理工学院、Facebook AI Research | ECCV 2020 | arXiv:2004.02857 |
| 03 | REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments | Yuankai Qi, Qi Wu, Peter Anderson, Xin Wang 等 | CVPR 2020 | arXiv:1904.10151 |

### 02 — 算法框架（7）

| # | 标题 | 机构 | 出处 | 链接 |
|---|------|------|------|------|
| 04 | Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training | 杜克大学、微软研究院 | CVPR 2020 | arXiv:2002.10638 |
| 05 | VLN↻BERT: A Recurrent Vision-and-Language BERT for Navigation | 澳大利亚国立大学、阿德莱德大学 | CVPR 2021 | arXiv:2011.13922 |
| 06 | History Aware Multimodal Transformer for Vision-and-Language Navigation | 法国国家信息与自动化研究所等 | NeurIPS 2021 | arXiv:2110.13309 |
| 07 | Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation | 法国国家信息与自动化研究所等 | CVPR 2022 | arXiv:2202.11742 |
| 08 | Scaling Data Generation in Vision-and-Language Navigation | 澳大利亚国立大学、上海 AI Lab | ICCV 2023 | arXiv:2307.15644 |
| 09 | ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments | 中国科学院大学 | IEEE TPAMI 2024 | arXiv:2304.03047 |
| 10 | NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation | 北京大学、BAAI 等 | RSS 2024 | arXiv:2402.15852 |

## 对 wiki 的映射

- [vln-10-papers-technology-map](../../wiki/overview/vln-10-papers-technology-map.md)（父节点 + Mermaid）
- [vln-category-01-datasets-platforms](../../wiki/overview/vln-category-01-datasets-platforms.md)、[vln-category-02-algorithm-frameworks](../../wiki/overview/vln-category-02-algorithm-frameworks.md)
- 论文实体：`wiki/entities/paper-vln-01-r2r.md` … `paper-vln-10-navid.md`
- **NaVid**（RSS 2024，arXiv:2402.15852）≠ **Uni-NaVid**（RSS 2025 导航 VLA 复现栈，见 [四范式复现路径](../../wiki/overview/vln-open-source-repro-paradigms.md)）

## 可信度与使用边界

- 本文为 **微信公众号策展导读**（「盘点」体例），论文细节以 arXiv / 原文 PDF 为准。
- Google Scholar 引用量为文内统计，会随时间变化。
- 原始抓取正文见 [wechat_vln_10_papers_2026-06-20.md](../raw/wechat_vln_10_papers_2026-06-20.md)。
