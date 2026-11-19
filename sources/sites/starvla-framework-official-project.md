# StarVLA 官方项目主页 / StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing

- **类型：** site / 官方项目主页
- **项目主页：** https://starvla.github.io/
- **官方论文页：** https://starvla.github.io/papers/starvla/
- **主论文：** https://arxiv.org/abs/2604.05014
- **论文全文：** https://arxiv.org/html/2604.05014v1
- **代码：** https://github.com/starVLA/starVLA
- **技术文档：** https://starvla.github.io/docs/overview/
- **开源状态：** 已开源（框架代码与模型/基准入口公开；不同分支、权重和训练配方需逐项核对）。
- **机构：** StarVLA Community、香港科技大学冯诺依曼研究院（Von Neumann Institute, HKUST）
- **核对日期：** 2026-11-19

## 项目页原始信息与源码开放核查

官方页面给出的积木结构是 **Backbone → Action head → Training recipe → Benchmark**；动作头包含 FAST、OFT、π/Flow、GR00T，共用训练与评测入口。网页中“切换配置”是框架的设计目标，不能推导出所有组合都有现成兼容 checkpoint、同样资源需求或同一成功率。

- **公开论文：** arXiv:2604.05014；另有独立的 [StarVLA-α（2604.11757）](https://arxiv.org/abs/2604.11757)，不能合并为一篇论文实体。
- **代码与文档：** 项目页直接链接官方 GitHub、教程、模型合集与 benchmark 入口。训练/推理可读源码已在 `starVLA_dev` 分支核查；官方 README 提醒开发分支可能不稳定。
- **RL 状态：** 官网列出 RL（RLinf）训练配方；原始 2026-04-06 论文曾标记为进行中，须按官方后续版本和 RLinf 教程辨别发布状态。
- **数据可信边界：** 单基准 RoboCasa-GR1 Table 5 报告 FAST 39.0%、OFT 48.8%（差 9.8 个百分点）；Table 9 的跨四基准联合训练 Generalist 报告 RoboCasa-GR1 57.3%。Table 9 Specialist OFT 53.8% 与 Table 5 48.8% 不一致，引用时保留表号。
- **复现实验：** 原作者实验数字；本次为官方网页、论文和源码静态核查，未自行执行 GPU 训练和仿真评测。

## 对 wiki 的映射

- [StarVLA 框架论文唯一详情节点](../../wiki/entities/paper-starvla-framework.md)
- [StarVLA-α 独立论文详情](../../wiki/entities/paper-starvla-alpha.md)
- [框架论文原文核对](../papers/starvla_framework_arxiv_2604_05014_full_reading.md)
- [官方源码归档](../repos/star_vla.md)
