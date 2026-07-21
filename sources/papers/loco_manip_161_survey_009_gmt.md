# GMT

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 009/161）

- **标题：** GMT: General Motion Tracking for Humanoid Whole-Body Control
- **类型：** paper
- **Loco-Manip 161 分类：** 01 运控基座与通用全身跟踪
- **机构：** UC San Diego、Simon Fraser University
- **项目页：** https://gmt-humanoid.github.io
- **arXiv：** <https://arxiv.org/abs/2506.14770>
- **代码：** <https://github.com/zixuan417/humanoid-general-motion-tracking>（部分开源：sim2sim + pretrained）
- **发表日期：** 2025年9月4日（公众号标注）；arXiv:2506.14770
- **入库日期：** 2026-06-26；**校正：** 2026-07-21（对齐正式论文摘录）
- **一句话说明（校正后）：** Adaptive Sampling + Motion MoE 训练 **单一统一** 人形全身跟踪策略（教师 PPO + 学生 DAgger），在 filtered AMASS+LAFAN1 与 Unitree G1 真机上验证。
- **权威摘录：** [`gmt_arxiv_2506_14770.md`](./gmt_arxiv_2506_14770.md)

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 01 运控基座与通用全身跟踪，编号 **009/161**。
- **算法实现总结（公众号原文，含误述，仅作溯源）：** 「扩散策略/流匹配…」——**与 arXiv 不符**，勿写入 wiki 方法定义。
- **校正要点：** GMT 是 **RL 运动跟踪**（Adaptive Sampling、Motion MoE、特权教师–学生），不是扩散动作生成器。

## 对 wiki 的映射

- 正式实体：[paper-gmt](../../wiki/entities/paper-gmt.md)
- 161 索引：[paper-loco-manip-161-009-gmt](../../wiki/entities/paper-loco-manip-161-009-gmt.md)
- [loco-manip-161-category-01-motion-base-wbt](../../wiki/overview/loco-manip-161-category-01-motion-base-wbt.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
- 论文 / 项目页 / 代码：见 [`gmt_arxiv_2506_14770.md`](./gmt_arxiv_2506_14770.md)、[`gmt-humanoid-github-io.md`](../sites/gmt-humanoid-github-io.md)、[`humanoid-general-motion-tracking.md`](../repos/humanoid-general-motion-tracking.md)
