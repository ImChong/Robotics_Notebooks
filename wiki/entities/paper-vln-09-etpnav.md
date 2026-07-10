---

type: entity
tags: [paper, vln, vln-survey, navigation, embodied-ai, ucas]
status: complete
updated: 2026-07-10
arxiv: "2304.03047"
summary: "连续环境 VLN-CE 下拓扑建图、跨模态规划与底层控制（含避障）三模块串联的端到端框架。"
related:
  - ../overview/vln-10-papers-technology-map.md
  - ../overview/vln-category-02-algorithm-frameworks.md
  - ../tasks/vision-language-navigation.md
  - ../entities/paper-realm-last-3-meter-vln-grounding.md
sources:
  - ../../sources/papers/vln_survey_09_etpnav.md
  - ../../sources/blogs/wechat_shenlan_vln_10_papers_survey.md
  - ../../sources/papers/vln_10_papers_catalog.md
  - ../../sources/papers/realm_last_3_meter_vln_arxiv_2607_03792.md
---

# ETPNav

**ETPNav** 收录于 [深蓝具身智能 · VLN 10 项代表性研究](https://mp.weixin.qq.com/s/2_dYaN6IeWn_vvS_jmGqRQ) **第 09/10** 篇，归类为 **02 算法框架**。

## 一句话定义

连续环境 VLN-CE 下拓扑建图、跨模态规划与底层控制（含避障）三模块串联的端到端框架。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-and-Language Navigation | 依据自然语言指令在环境中导航的具身任务 |
| R2R | Room-to-Room | Matterport3D 上经典逐步导航指令数据集 |
| VLM | Vision-Language Model | 视觉-语言多模态大模型，NaVid 等路线的骨干 |

## 为什么重要

- 连续环境拓扑规划代表基线，在 R2R-CE、RxR-CE 上报告大幅性能提升。
- **末段精修上游：** [REALM](../entities/paper-realm-last-3-meter-vln-grounding.md) 将 ETPNav-ZS/FT 作为四类 plug-and-play 骨干之一，在 REVERIE-AIM 上 ONS@0.1m 相对无精修基线约 **翻倍**（arXiv:2607.03792）。
- 连续环境 VLN-CE 下拓扑建图、跨模态规划与底层控制（含避障）三模块串联的端到端框架。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 09/10 |
| 分组 | 02 算法框架 |
| 出处 | IEEE TPAMI 2024 · arXiv:2304.03047 |
| 机构 | 中国科学院大学 |

## 核心机制（归纳）

### 1）策展导读要点

**任务形式：** 连续环境 VLN-CE 下拓扑建图、跨模态规划与底层控制（含避障）三模块串联的端到端框架。

### 2）策展导读要点

**机构/出处：** 中国科学院大学 · IEEE TPAMI 2024

### 3）策展导读要点

**在 VLN 地图中的位置：** 02 算法框架（#09/10）。

## 常见误区

1. VLN benchmark 提升不等于真机部署；连续环境 (VLN-CE) 与离散图设定不可直接混比。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 技术地图：[vln-10-papers-technology-map.md](../overview/vln-10-papers-technology-map.md)
- 分类 hub：[vln-category-02-algorithm-frameworks.md](../overview/vln-category-02-algorithm-frameworks.md)
- 任务页：[vision-language-navigation.md](../tasks/vision-language-navigation.md)
- 末段精修：[REALM](../entities/paper-realm-last-3-meter-vln-grounding.md) — ETPNav 为 REALM 主要上游骨干
- 原始 source：[vln_survey_09_etpnav.md](../../sources/papers/vln_survey_09_etpnav.md)

## 参考来源

- [vln_survey_09_etpnav.md](../../sources/papers/vln_survey_09_etpnav.md) — VLN 10 篇策展摘录
- [vln_10_papers_catalog.md](../../sources/papers/vln_10_papers_catalog.md)
- [wechat_shenlan_vln_10_papers_survey.md](../../sources/blogs/wechat_shenlan_vln_10_papers_survey.md)
- 论文：<https://arxiv.org/abs/2304.03047>

## 推荐继续阅读

- [VLN 10 篇技术地图](../overview/vln-10-papers-technology-map.md)
- [视觉–语言导航（VLN）](../tasks/vision-language-navigation.md)
