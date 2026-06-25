# TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size

> 来源归档（ingest · 人形 AMP 运动先验 19 篇 · 第 17/19）

- **标题：** TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size
- **类型：** paper
- **AMP 叙事段：** 04 交互与长时程
- **venue：** IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR）2026
- **机构：** Garena、Sea AI Lab、NUS
- **论文链接：** <https://arxiv.org/abs/2603.07988>
- **项目页：** <https://splionar.github.io/TeamHOI/>
- **代码：** <https://github.com/sail-sg/TeamHOI>
- **索引来源：** [具身智能研究室 · AMP 专题长文](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)（<https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w>）
- **原始抓取：** [wechat_humanoid_amp_19_survey_2026-05-26.md](../raw/wechat_humanoid_amp_19_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26
- **深读归档：** 2026-06-25 — [teamhoi_arxiv_2603_07988.md](teamhoi_arxiv_2603_07988.md)
- **一句话说明：** 单一去中心化 Transformer + masked AMP，2–8 人协作搬桌单策略泛化。

## 核心摘录（策展，非全文）

- **在 AMP 四段地图中的位置：** 04 交互与长时程，编号 **17/19**。
- **公众号导读要点：** 多 humanoid 协作搬运；缺协作 MoCap 时用单人参考 + mask。
- **方法要点（arXiv 编译）：** 队友 token cross-attention；$D_{\text{full}}$/$D_{\text{mask}}$ 混合；formation 奖励队形/形状无关。

## 对 wiki 的映射

- [paper-amp-survey-17-teamhoi](../../wiki/entities/paper-amp-survey-17-teamhoi.md) — 主实体页
- [amp-reward](../../wiki/methods/amp-reward.md)
- [PhysHSI #15](../../wiki/entities/paper-amp-survey-15-physhsi.md) — 单人 HSI 对照
- [MoRE #08](../../wiki/entities/paper-amp-survey-08-more.md)、[Goalkeeper #13](../../wiki/entities/paper-amp-survey-13-humanoid_goalkeeper.md) — 条件化/分部位 AMP 家族

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
- 姊妹篇 42 篇栈：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- arXiv 深读：[teamhoi_arxiv_2603_07988.md](teamhoi_arxiv_2603_07988.md)
