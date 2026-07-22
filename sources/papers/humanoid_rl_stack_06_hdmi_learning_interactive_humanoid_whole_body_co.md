# HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos

> 来源归档（ingest · 人形 RL 身体系统栈 42 篇 · 第 06/42）

- **标题：** HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos
- **类型：** paper
- **系统栈层：** 01 数据 · 重定向 · 遥操作
- **机构：** CMU
- **项目/论文链接：** <https://hdmi-humanoid.github.io>
- **arXiv：** <https://arxiv.org/abs/2509.16757>
- **代码：** <https://github.com/LeCAR-Lab/HDMI>
- **索引来源：** [具身智能研究室 · 42 篇 RL 运动控制长文](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）
- **原始抓取：** [wechat_humanoid_rl_42_survey_2026-05-26.md](../raw/wechat_humanoid_rl_42_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26
- **复核日期：** 2026-07-22
- **一句话说明：** HDMI 的全称是 HumanoiD iMitation for Interaction。它也从人类视频出发，但比 HumanX 更进一步，把重点放在 contact-rich humanoid-object interaction 上。

## 核心摘录（策展，非全文）

- **在身体系统栈中的位置：** 01 数据 · 重定向 · 遥操作，编号 **06/42**。
- **公众号导读要点：** HDMI 的全称是 HumanoiD iMitation for Interaction。它也从人类视频出发，但比 HumanX 更进一步，把重点放在 contact-rich humanoid-object interaction 上。
- **读者动作：** 方法细节以论文 PDF / 项目页为准；总框架见 [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)。
- **论文复核要点：** HDMI 从单目 RGB 视频恢复人体与物体轨迹，构建 `motion.npz` + `meta.json` 格式参考；RL 阶段用 robot-object co-tracking、统一物体表示、残差动作空间和接触奖励训练 G1 策略。
- **开源状态：** 官方 [LeCAR-Lab/HDMI](https://github.com/LeCAR-Lab/HDMI) 已开放 IsaacSim / IsaacLab 训练代码；sim2real 细节另指向 `EGalahad/sim2real`。

## 对 wiki 的映射

  - [HDMI 论文实体](../../wiki/entities/paper-hrl-stack-06-hdmi.md)
  - [HDMI 项目页归档](../sites/hdmi-project.md)
  - [HDMI 官方代码归档](../repos/hdmi.md)
  - [Loco-Manip 接触分类 01：接触数据](../../wiki/overview/loco-manip-contact-category-01-contact-data.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- 姊妹篇 AMP 专题：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
- 项目页：<https://hdmi-humanoid.github.io>
- arXiv：<https://arxiv.org/abs/2509.16757>
- 代码：<https://github.com/LeCAR-Lab/HDMI>
