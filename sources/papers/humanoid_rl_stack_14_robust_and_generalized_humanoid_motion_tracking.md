# Robust and Generalized Humanoid Motion Tracking

> 来源归档（ingest · 人形 RL 身体系统栈 42 篇 · 第 14/42）

- **标题：** Robust and Generalized Humanoid Motion Tracking
- **类型：** paper
- **系统栈层：** 02 参考跟踪 · 通用控制
- **机构：** 北京理工大学；人形机器人（上海）有限公司
- **项目/论文链接：** <https://zeonsunlightyu.github.io/RGMT.github.io/>
- **索引来源：** [具身智能研究室 · 42 篇 RL 运动控制长文](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）
- **原始抓取：** [wechat_humanoid_rl_42_survey_2026-05-26.md](../raw/wechat_humanoid_rl_42_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26
- **一句话说明：** RGMT 是 robust and generalized humanoid motion tracking。它的核心是 dynamics-conditioned aggregation：用 causal temporal encoder 总结近期本体状态，用 multi-head command encoder 选择性聚合参考命令。论文还设计 recovery curriculum 和 annealed upward assistance force 来增强恢复和抗扰。

## 核心摘录（策展，非全文）

- **在身体系统栈中的位置：** 02 参考跟踪 · 通用控制，编号 **14/42**。
- **公众号导读要点：** RGMT 是 robust and generalized humanoid motion tracking。它的核心是 dynamics-conditioned aggregation：用 causal temporal encoder 总结近期本体状态，用 multi-head command encoder 选择性聚合参考命令。论文还设计 recovery curriculum 和 annealed upward assistance force 来增强恢复和抗扰。
- **读者动作：** 方法细节以论文 PDF / 项目页为准；总框架见 [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)。

## 对 wiki 的映射

  - [RGMT 实体页](../../wiki/entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md)
  - 后续扩展：[Extreme-RGMT](../../wiki/entities/paper-extreme-rgmt.md)（arXiv:2607.20110）
  - 方法页：[Any2Track & RGMT](../../wiki/methods/any2track.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- 姊妹篇 AMP 专题：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
