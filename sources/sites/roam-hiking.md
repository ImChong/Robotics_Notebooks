# ROAM — 官方项目主页

- **类型：** site（论文项目主页）
- **项目：** [ROAM: Robust Open-world Autonomous Mobility](https://roam-hiking.github.io/)
- **论文：** [Bridging vision-language reasoning and learned control for legged navigation in the wild](https://doi.org/10.1126/scirobotics.aee1549)（[归档](../papers/roam_legged_navigation_arxiv_2611_25215.md)）
- **arXiv：** [2611.25215](https://arxiv.org/abs/2611.25215)
- **代码：** [leggedrobotics/roam](https://github.com/leggedrobotics/roam)（[归档](../repos/leggedrobotics-roam.md)）
- **入库日期：** 2026-11-20
- **核对范围：** 直接核查官方页面 How it works、山地路线交互、Signs、Locomotion、Navigation、视频、参考文献及 Code 入口。
- **通读：** 已完成

## 原文事实

- 主线：VLM 以较低频率评估语义与 `risk/steppable/preferred_area`，SAN 高速分割、语义高程图融合、RL navigation 和 locomotion 独立高频响应。
- 官方交互给出 Zermatt 10.1 km / 1220 m 下降（2024-08-14/15）、Seealpsee 2.38 km（2024-06-25）两次长程实地记录；实验日期不同于 2026 期刊发布日期。
- 页面给出真实场景风险图、标志牌语义对照、坡地与金属罐落足示例和开放词汇分割对比。
- 官方 Code 链接可访问，但主仓 README 明确只覆盖 VLM 分析/开放词汇分割及与 elevation_mapping_cupy 的连接；完整真机策略运行不是此仓提供的复现范围。

## 对 wiki 的映射

- [ROAM 论文唯一详情节点](../../wiki/entities/paper-roam-open-world-legged-navigation.md)
