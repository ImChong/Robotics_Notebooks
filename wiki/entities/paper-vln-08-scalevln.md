---
type: entity
tags: [paper, vln, vln-survey, navigation, embodied-ai]
status: complete
updated: 2026-06-20
arxiv: "2307.15644"
summary: "从 HM3D/Gibson 1200+ 场景自动采样路径、风格迁移与指令生成，构建 490 万指令-轨迹对预训练数据。"
related:
  - ../overview/vln-10-papers-technology-map.md
  - ../overview/vln-category-02-algorithm-frameworks.md
  - ../tasks/vision-language-navigation.md
sources:
  - ../../sources/papers/vln_survey_08_scalevln.md
  - ../../sources/blogs/wechat_shenlan_vln_10_papers_survey.md
  - ../../sources/papers/vln_10_papers_catalog.md
---

# ScaleVLN

**ScaleVLN** 收录于 [深蓝具身智能 · VLN 10 项代表性研究](https://mp.weixin.qq.com/s/2_dYaN6IeWn_vvS_jmGqRQ) **第 08/10** 篇，归类为 **02 算法框架**。本页为知识库 **策展摘要**；方法细节以论文 PDF 为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-and-Language Navigation | 依据自然语言指令在环境中导航的具身任务 |
| R2R | Room-to-Room | Matterport3D 上经典逐步导航指令数据集 |
| VLM | Vision-Language Model | 视觉-语言多模态大模型，NaVid 等路线的骨干 |

## 为什么重要

- 证明数据规模扩展可显著抬升 R2R 等 benchmark 上限（文中 R2R SR 约 69%→80%）。
- 在 [VLN 10 篇技术地图](../overview/vln-10-papers-technology-map.md) 中属于 **[02 算法框架](../overview/vln-category-02-algorithm-frameworks.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 08/10 |
| 分组 | 02 算法框架 |
| 出处 | ICCV 2023 · arXiv:2307.15644 |
| 机构 | 澳大利亚国立大学、上海 AI Lab |

## 与其他页面的关系

- 技术地图：[vln-10-papers-technology-map.md](../overview/vln-10-papers-technology-map.md)
- 分类 hub：[vln-category-02-algorithm-frameworks.md](../overview/vln-category-02-algorithm-frameworks.md)
- 任务页：[vision-language-navigation.md](../tasks/vision-language-navigation.md)
- 原始 source：[vln_survey_08_scalevln.md](../../sources/papers/vln_survey_08_scalevln.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF** 为准。

## 参考来源

- [vln_survey_08_scalevln.md](../../sources/papers/vln_survey_08_scalevln.md) — VLN 10 篇策展摘录
- [vln_10_papers_catalog.md](../../sources/papers/vln_10_papers_catalog.md)
- [wechat_shenlan_vln_10_papers_survey.md](../../sources/blogs/wechat_shenlan_vln_10_papers_survey.md)
- 论文：<https://arxiv.org/abs/2307.15644>

## 推荐继续阅读

- [VLN 10 篇技术地图](../overview/vln-10-papers-technology-map.md)
- [视觉–语言导航（VLN）](../tasks/vision-language-navigation.md)
