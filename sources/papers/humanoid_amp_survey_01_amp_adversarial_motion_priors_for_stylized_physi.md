# AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control

> 来源归档（ingest · 人形 AMP 运动先验 19 篇 · 第 01/19）

- **标题：** AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control
- **类型：** paper
- **AMP 叙事段：** 01 分布约束与先验组件化
- **机构：** UC Berkeley、上海交通大学
- **出处：** SIGGRAPH 2021
- **论文链接：** <https://xbpeng.com/projects/AMP/index.html>
- **索引来源：** [具身智能研究室 · AMP 专题长文](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)（<https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w>）
- **原始抓取：** [wechat_humanoid_amp_19_survey_2026-05-26.md](../raw/wechat_humanoid_amp_19_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26
- **深读归档：** 2026-06-25 — [amp.md](amp.md)
- **一句话说明：** AMP 的核心不是让角色逐帧复现某段参考动作，而是让策略生成的状态转移尽量接近动作数据里的状态转移。

## 核心摘录（策展，非全文）

- **在 AMP 四段地图中的位置：** 01 分布约束与先验组件化，编号 **01/19**。
- **公众号导读要点：** AMP 的核心不是让角色逐帧复现某段参考动作，而是让策略生成的状态转移尽量接近动作数据里的状态转移。
- **方法要点（深读编译）：** 短窗口状态转移判别 + 任务/风格复合 PPO；相对 DeepMimic 允许偏离参考轨迹完成新任务；后人形 SD-AMP、MoRE、AMP_mjlab 等均在此范式上扩展。

## 对 wiki 的映射

- [paper-amp-survey-01-amp](../../wiki/entities/paper-amp-survey-01-amp.md) — 主实体页（深读归纳）
- [amp-reward](../../wiki/methods/amp-reward.md)、[add](../../wiki/methods/add.md)
- [humanoid-amp-motion-prior-survey](../../wiki/overview/humanoid-amp-motion-prior-survey.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
- 姊妹篇 42 篇栈：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- 深读：[amp.md](amp.md)
