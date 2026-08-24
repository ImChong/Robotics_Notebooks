# T-GMP 地形条件生成式运动先验的人形自然多地形行走

> 来源归档（ingest · 运动小脑 64 篇长文 第 02/64；2026-08-24 补全 arXiv 深读）

- **标题：** T-GMP: Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion
- **类型：** paper
- **运动小脑分类：** A 走路底座
- **arXiv：** <https://arxiv.org/abs/2606.06944>
- **机构：** 哈尔滨工业大学（HIT）；乐聚机器人（Leju Robotics）
- **项目页：** <https://t-gmp.github.io>
- **入库日期：** 2026-06-18（策展索引）；2026-08-24（arXiv 全文 ingest）
- **一句话说明：** 地形条件 **CVAE 生成式运动先验** + **地形条件 AMP 判别器** + **Foothold Penalty**，在乐聚 **Kuavo** 上实现八地形统一策略与自然全身协调（摆臂、降质心、伸臂平衡）。

## 核心摘录（策展 + 论文）

- **在动作小脑地图中的位置：** A 走路底座，编号 **02/64**。
- **方法主干：** 特权专家 + MoCap（GMR）→ 离线 T-GMP（条件 β-VAE）→ 地形条件判别器注入 $r_{\mathrm{amp}}$ → PPO 训练 Actor（height scan + 本体历史）。
- **数据效率：** 约 **29.6 min / 88.8k 帧** 专家数据覆盖八地形流形。
- **开源结论：** 项目页 **404**；截至 2026-08-24 **无官方 GitHub** — 见 [`sources/sites/t-gmp.md`](../sites/t-gmp.md)。

## 对 wiki 的映射

- [paper-motion-cerebellum-t-gmp](../../wiki/entities/paper-motion-cerebellum-t-gmp.md)
- [motion-cerebellum-category-01-locomotion-base](../../wiki/overview/motion-cerebellum-category-01-locomotion-base.md)
- 深读归档：[t_gmp_terrain_conditioned_generative_motion_priors_arxiv_2606_06944.md](t_gmp_terrain_conditioned_generative_motion_priors_arxiv_2606_06944.md)

## 参考来源（原始）

- arXiv:2606.06944
- 项目页：https://t-gmp.github.io
- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
