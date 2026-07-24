# Extreme-RGMT 项目页（zeonsunlightyu.github.io）

- **标题：** Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust Generalist Humanoid Control
- **类型：** site / project-page
- **URL：** <https://zeonsunlightyu.github.io/Extreme-RGMT.github.io/>
- **论文：** [arXiv:2607.20110](https://arxiv.org/abs/2607.20110)（PDF：<https://arxiv.org/pdf/2607.20110>）
- **代码：** 截至 **2026-07-24** 复核，项目页仍仅有 arXiv PDF badge 与 BibTeX，**未列 GitHub / Hugging Face 仓库或权重链接**
- **入库日期：** 2026-07-23（2026-07-24 复核开源状态无变化）
- **配套论文归档：** [`sources/papers/extreme_rgmt_arxiv_2607_20110.md`](../papers/extreme_rgmt_arxiv_2607_20110.md)

## 一句话摘要

北京理工大学 / 人形机器人（上海）有限公司（青龙 OpenLoong）/ 山东大学提出的 **两阶段 continual learning** 全身 motion tracking 框架：Stage I 学 generalist 基座策略，Stage II 用 **PACE**（acquisition–consolidation）与 **STAR**（高优势片段重采样）在保留日常跟踪能力的同时扩展高动态技能；项目页以高动态 / 低动态 / 失败对比视频为主入口。

## 开源状态（步骤 2.5，截至 2026-07-24）

| 项 | 状态 |
|----|------|
| 项目页 Code / GitHub badge | **无** |
| Hugging Face / 权重 | **无** |
| 数据集下载链 | **无**（正文使用 LAFAN1 / AMASS / 自采 Xsens；评测引用 OmniXtreme 的 XtremeMotion） |
| 论文承诺 | 摘要与正文以项目页为入口，**未写 “code will be released” 明确 URL** |

结论：**确认未开源**（以项目页实际链接为准）。勿在 wiki 中暗示可复现训练代码。

## 页面内容要点

- **Abstract**：generalist vs specialist 权衡；PACE + STAR；固定参考与在线惯性 MoCap 双模态。
- **视频分区：** High Dynamic Motion（20）、Low Dynamic Motion（8）、Failed Motion（3）。
- **BibTeX：** `arXiv:2607.20110`，作者 Ma / Yu / Guo / Lv / Mao / Xing / Ren / Zheng。

## 关联资料

- 论文归档：[`sources/papers/extreme_rgmt_arxiv_2607_20110.md`](../papers/extreme_rgmt_arxiv_2607_20110.md)
- 前作 RGMT 项目页：<https://zeonsunlightyu.github.io/RGMT.github.io/>
- 前作 wiki：[`wiki/entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md`](../../wiki/entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md)
