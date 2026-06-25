# Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints

> 来源归档（ingest · 人形 AMP 运动先验 19 篇 · 第 13/19）

- **标题：** Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints
- **类型：** paper
- **AMP 叙事段：** 04 交互与长时程
- **机构：** 香港大学、上海人工智能实验室
- **论文链接：** <https://arxiv.org/abs/2510.18002>
- **项目页：** <https://humanoid-goalkeeper.github.io/Goalkeeper/>
- **代码：** <https://github.com/InternRobotics/Humanoid-Goalkeeper>
- **索引来源：** [具身智能研究室 · AMP 专题长文](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)（<https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w>）
- **原始抓取：** [wechat_humanoid_amp_19_survey_2026-05-26.md](../raw/wechat_humanoid_amp_19_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26
- **深读归档：** 2026-06-25 — [humanoid_goalkeeper_arxiv_2510_18002.md](humanoid_goalkeeper_arxiv_2510_18002.md)
- **一句话说明：** 单阶段端到端 PPO：球落点区域条件化任务奖 + 多判别器软 AMP，G1 宽范围自然扑救；MoCap/机载相机双模态。

## 核心摘录（策展，非全文）

- **在 AMP 四段地图中的位置：** 04 交互与长时程，编号 **13/19**。
- **公众号导读要点：** 运动先验用于守门任务中的自然全身反应；端到端 RL 替代遥操作/固定跟踪。
- **方法要点（arXiv 编译）：** 6 区域 position-conditioned 任务/AMP；GVHMR 参考 + 软高斯 AMP；仿真 $E_{\text{succ}}\approx 81\%$；真机 MoCap 21/30。

## 对 wiki 的映射

- [paper-amp-survey-13-humanoid_goalkeeper](../../wiki/entities/paper-amp-survey-13-humanoid_goalkeeper.md) — 主实体页（深读归纳）
- [amp-reward](../../wiki/methods/amp-reward.md)、[unitree-g1](../../wiki/entities/unitree-g1.md)
- [MoRE #08](../../wiki/entities/paper-amp-survey-08-more.md) — 多判别器条件化先验对照
- [PhysHSI #15](../../wiki/entities/paper-amp-survey-15-physhsi.md)、[HUSKY #14](../../wiki/entities/paper-amp-survey-14-husky.md) — 同段交互姊妹篇

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
- 姊妹篇 42 篇栈：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- arXiv 深读：[humanoid_goalkeeper_arxiv_2510_18002.md](humanoid_goalkeeper_arxiv_2510_18002.md)
