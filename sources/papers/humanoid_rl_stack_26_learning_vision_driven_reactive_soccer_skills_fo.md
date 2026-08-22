# Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots

> 来源归档（ingest · 人形 RL 身体系统栈 42 篇 · 第 26/42）

- **标题：** Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots
- **类型：** paper
- **系统栈层：** 03 感知式高动态运动
- **机构：** 清华大学；字节跳动 Seed；中国农业大学
- **项目/论文链接：** <https://humanoid-kick.github.io>
- **arXiv：** <https://arxiv.org/abs/2511.03996>（v2，2026-08-20）
- **正式发表：** [Science Robotics 11, eaed1152 (2026)](https://doi.org/10.1126/scirobotics.aed1152)
- **索引来源：** [具身智能研究室 · 42 篇 RL 运动控制长文](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）
- **原始抓取：** [wechat_humanoid_rl_42_survey_2026-05-26.md](../raw/wechat_humanoid_rl_42_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26
- **再核日期：** 2026-08-22
- **一句话说明：** 统一 RL 控制器耦合视觉与运动；虚拟感知 + encoder-decoder 恢复特权态；AMP 运动先验；机载视觉实现寻球/追球/多向踢球（清华 / 字节 Seed / 农大）。
- **开源状态（2026-08-22）：** **部分开源** — Zenodo [21620490](https://zenodo.org/records/21620490) 发布 `code.zip`（Isaac Gym 训练 + 推理 + checkpoint）；**无 GitHub**；真机部署未随包发布。

## 核心摘录（策展，非全文）

- **在身体系统栈中的位置：** 03 感知式高动态运动，编号 **26/42**。
- **问题：** 模块化感知–控制流水线在真实视觉噪声下难保持反应式连贯行为。
- **方法：** 虚拟感知系统建模机载视觉误差；encoder-decoder 从历史部分观测恢复状态；PPO + AMP 判别器 + 多 critic；部署侧相机球检测 + 里程计门位。
- **摘要量化：** 相对规则基线，球位估计误差 **−46%**、time-to-kick **−64%**；前场踢球成功率约 **90%**；室外、动态场景与真实 RoboCup 比赛验证。
- **读者动作：** 方法细节以 Science Robotics / arXiv PDF 为准；仿真复现见 Zenodo `code.zip`；总框架见 [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)。

## 对 wiki 的映射

- [paper-hrl-stack-26-learning_vision_driven_reactive_socc](../../wiki/entities/paper-hrl-stack-26-learning_vision_driven_reactive_socc.md)
- [humanoid-kick-vision-driven-soccer.md](../sites/humanoid-kick-vision-driven-soccer.md)
- [humanoid-kick-vision-driven-soccer.md](../repos/humanoid-kick-vision-driven-soccer.md)
- [humanoid-soccer](../../wiki/tasks/humanoid-soccer.md)

## 参考来源（原始）

- 论文 PDF：<https://arxiv.org/pdf/2511.03996>
- 项目页：<https://humanoid-kick.github.io>
- 代码包：<https://zenodo.org/records/21620490>
- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- 姊妹篇 AMP 专题：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
