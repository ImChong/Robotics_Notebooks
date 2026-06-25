# MoRE: Mixture of Residual Experts for Humanoid Lifelike Gaits Learning on Complex Terrains

> 来源归档（ingest · 人形 AMP 运动先验 19 篇 · 第 08/19）

- **标题：** MoRE: Mixture of Residual Experts for Humanoid Lifelike Gaits Learning on Complex Terrains
- **类型：** paper
- **AMP 叙事段：** 02 人形走跑
- **机构：** 中科大、中国电信人工智能研究院、哈尔滨工程大学、上海科技大学
- **论文链接：** <https://arxiv.org/abs/2506.08840>
- **项目页：** <https://more-humanoid.github.io/>
- **索引来源：** [具身智能研究室 · AMP 专题长文](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)（<https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w>）
- **原始抓取：** [wechat_humanoid_amp_19_survey_2026-05-26.md](../raw/wechat_humanoid_amp_19_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26
- **深读归档：** 2026-06-25 — [more_mixture_residual_experts_arxiv_2506_08840.md](more_mixture_residual_experts_arxiv_2506_08840.md)
- **一句话说明：** 两阶段：深度相机 base locomotion + latent residual MoE 与多判别器 AMP，在复杂地形用 gait command 切换多种人形步态。

## 核心摘录（策展，非全文）

- **在 AMP 四段地图中的位置：** 02 人形走跑，编号 **08/19**。
- **公众号导读要点：** 自然步态不是固定风格，而是平地/坡道/跨障时可切换的运动模式；MoRE 用多 expert 让「像人」具备状态/命令依赖；AMP 不能只做平地美化，必须进入地形与步态切换。
- **方法要点（arXiv 编译）：** Stage 1 无 motion prior 学深度条件穿越；Stage 2 MoE 残差 + 每步态一个 AMP 判别器（Walk-Run / High-Knees / Squat）+ gait rewards；G1 + RealSense D435i 真机 50 Hz。

## 对 wiki 的映射

- [paper-amp-survey-08-more](../../wiki/entities/paper-amp-survey-08-more.md) — 主实体页（深读归纳）
- [amp-reward](../../wiki/methods/amp-reward.md)、[locomotion](../../wiki/tasks/locomotion.md)、[terrain-adaptation](../../wiki/concepts/terrain-adaptation.md)
- [unitree-g1](../../wiki/entities/unitree-g1.md)、[lafan1-dataset](../../wiki/entities/lafan1-dataset.md)
- [paper-explicit-stair-geometry-humanoid-locomotion](../../wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md) — 楼梯任务中将 MoRE 作视觉基线对照

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
- 姊妹篇 42 篇栈：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- arXiv 深读：[more_mixture_residual_experts_arxiv_2506_08840.md](more_mixture_residual_experts_arxiv_2506_08840.md)
