---
type: overview
tags: [vln, navigation, algorithms, transformer, vlm, category-hub, survey]
status: complete
updated: 2026-06-20
summary: "VLN 10 篇盘点 · 02 算法框架（7 篇）— 预训练对齐、历史记忆、拓扑规划、数据扩展与 VLM 端到端如何演进？"
related:
  - ./vln-10-papers-technology-map.md
  - ./vln-category-01-datasets-platforms.md
  - ../entities/paper-vln-04-prevalent.md
  - ../entities/paper-vln-05-vln-bert.md
  - ../entities/paper-vln-06-hamt.md
  - ../entities/paper-vln-07-duet.md
  - ../entities/paper-vln-08-scalevln.md
  - ../entities/paper-vln-09-etpnav.md
  - ../entities/paper-vln-10-navid.md
  - ../tasks/vision-language-navigation.md
  - ./vln-open-source-repro-paradigms.md
sources:
  - ../../sources/blogs/wechat_shenlan_vln_10_papers_survey.md
  - ../../sources/papers/vln_10_papers_catalog.md
---

# VLN 分类 02：算法框架

> **图谱分类节点**：对应 [深蓝具身智能 · VLN 10 项代表性研究](https://mp.weixin.qq.com/s/2_dYaN6IeWn_vvS_jmGqRQ) 的 **02 算法框架** 分组；总地图见 [VLN 10 篇技术地图](./vln-10-papers-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-and-Language Navigation | 依据自然语言在环境中导航的具身任务 |
| VLM | Vision-Language Model | 视觉-语言多模态大模型，NaVid 等路线的骨干 |
| VLN-CE | VLN in Continuous Environments | 连续动作空间下的 VLN 设定与 benchmark 族 |

## 核心问题

**VLN 模型如何从「从头训练」走向「预训练 + 记忆 + 规划 + 扩数据 + VLM」？** 本组 7 篇覆盖预训练范式（PREVALENT）、循环/完整历史建模（VLN↻BERT、HAMT）、双尺度拓扑规划（DUET、ETPNav）、大规模数据生成（ScaleVLN）与视频流 VLM 端到端（NaVid）。

## 本组论文（7 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 04 | PREVALENT | [paper-vln-04-prevalent.md](../entities/paper-vln-04-prevalent.md) | [source](../../sources/papers/vln_survey_04_prevalent.md) |
| 05 | VLN↻BERT | [paper-vln-05-vln-bert.md](../entities/paper-vln-05-vln-bert.md) | [source](../../sources/papers/vln_survey_05_vln_bert.md) |
| 06 | HAMT | [paper-vln-06-hamt.md](../entities/paper-vln-06-hamt.md) | [source](../../sources/papers/vln_survey_06_hamt.md) |
| 07 | DUET | [paper-vln-07-duet.md](../entities/paper-vln-07-duet.md) | [source](../../sources/papers/vln_survey_07_duet.md) |
| 08 | ScaleVLN | [paper-vln-08-scalevln.md](../entities/paper-vln-08-scalevln.md) | [source](../../sources/papers/vln_survey_08_scalevln.md) |
| 09 | ETPNav | [paper-vln-09-etpnav.md](../entities/paper-vln-09-etpnav.md) | [source](../../sources/papers/vln_survey_09_etpnav.md) |
| 10 | NaVid | [paper-vln-10-navid.md](../entities/paper-vln-10-navid.md) | [source](../../sources/papers/vln_survey_10_navid.md) |

## 演进脉络（策展）

| 阶段 | 代表 | 核心贡献 |
|------|------|----------|
| 预训练范式 | PREVALENT | 图像-文本-动作三元组 Masked LM + 动作预测自监督 |
| 历史建模 | VLN↻BERT → HAMT | 状态 token 压缩 → 完整历史 ViT 编码 |
| 拓扑规划 | DUET → ETPNav | 离散图双尺度规划 → 连续环境三模块串联 |
| 数据扩展 | ScaleVLN | HM3D/Gibson 自动采样 + 490 万指令-轨迹对 |
| VLM 融合 | NaVid | 视频流 + Vicuna-7B，无需地图/深度/里程计 |

## 与数据/平台组的分工

| 维度 | [01 数据/平台](./vln-category-01-datasets-platforms.md) | 本组（02 算法框架） |
|------|--------------------------------------------------------|---------------------|
| 回答 | **测什么、在哪测、动作空间如何定义** | **模型如何对齐、记忆、规划、扩数据、接 VLM** |
| 代表 | R2R / VLN-CE / REVERIE | PREVALENT → NaVid |

## NaVid 与 Uni-NaVid 的边界

- **[NaVid](../entities/paper-vln-10-navid.md)**（RSS 2024）：本文盘点第 10 篇，Vicuna-7B 视频流导航。
- **Uni-NaVid**（RSS 2025）：[四范式复现路径](./vln-open-source-repro-paradigms.md) 中的导航 VLA 开源栈，**不同论文、不同仓库**。

## 关联页面

- [VLN 10 篇技术地图](./vln-10-papers-technology-map.md)
- [数据集与仿真平台](./vln-category-01-datasets-platforms.md)
- [视觉–语言导航（VLN）](../tasks/vision-language-navigation.md)
- [VLN 四范式开源复现路径](./vln-open-source-repro-paradigms.md)

## 参考来源

- [wechat_shenlan_vln_10_papers_survey.md](../../sources/blogs/wechat_shenlan_vln_10_papers_survey.md)
- [vln_10_papers_catalog.md](../../sources/papers/vln_10_papers_catalog.md)

## 推荐继续阅读

- [VLN 四范式开源复现路径](./vln-open-source-repro-paradigms.md)
