# WholeBodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control

> 来源归档（ingest · 人形 RL 身体系统栈 42 篇 · 第 30/42）

- **标题：** WholeBodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control
- **类型：** paper
- **系统栈层：** 04 视觉闭环 · 任务接口 · 世界模型
- **机构：** 复旦大学；OpenDriveLab & 香港大学 MMLab；智元机器人；SII
- **项目/论文链接：** <https://opendrivelab.com/WholeBodyVLA>
- **索引来源：** [具身智能研究室 · 42 篇 RL 运动控制长文](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）
- **原始抓取：** [wechat_humanoid_rl_42_survey_2026-05-26.md](../raw/wechat_humanoid_rl_42_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26
- **一句话说明：** WholeBodyVLA 讨论的是全身 loco-manipulation VLA。它关注的问题是：人形机器人在大空间里完成抓取、搬运、推车等任务时，locomotion 和 manipulation 不能被简单拆开。

## 核心摘录（策展，非全文）

- **在身体系统栈中的位置：** 04 视觉闭环 · 任务接口 · 世界模型，编号 **30/42**。
- **公众号导读要点：** WholeBodyVLA 讨论的是全身 loco-manipulation VLA。它关注的问题是：人形机器人在大空间里完成抓取、搬运、推车等任务时，locomotion 和 manipulation 不能被简单拆开。
- **读者动作：** 方法细节以论文 PDF / 项目页为准；总框架见 [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)。

## 对 wiki 的映射

- [paper-hrl-stack-30-wholebodyvla](../../wiki/entities/paper-hrl-stack-30-wholebodyvla.md)
- [humanoid-rl-motion-control-body-system-stack](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)
- [loco-manip-contact-category-05-vla-world-models](../../wiki/overview/loco-manip-contact-category-05-vla-world-models.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- 姊妹篇 AMP 专题：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)

## 项目页与开源状态核查（2026-07-22）

- **论文/项目：** ICLR 2026, arXiv:2512.11047；项目页 <https://opendrivelab.com/WholeBodyVLA>。
- **GitHub：** <https://github.com/OpenDriveLab/WholebodyVLA>，MIT；README 明确当前无 open-source codebase 时间表，仓库主要是资源/参考集合。
- **关键数字：** latent action 约 10 Hz 解码；LMO policy 50 Hz 执行；Agibot X2 cart pushing 负载超过 50 kg。
- **wiki 深化：** [paper-hrl-stack-30-wholebodyvla](../../wiki/entities/paper-hrl-stack-30-wholebodyvla.md)。
