# DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos

> 来源归档（ingest · 人形 RL 身体系统栈 42 篇 · 第 35/42）

- **标题：** DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos
- **类型：** paper
- **系统栈层：** 04 视觉闭环 · 任务接口 · 世界模型
- **机构：** NVIDIA；香港科技大学；伯克利；华盛顿大学；斯坦福大学；KAIST；德州大学奥斯汀分校等
- **项目/论文链接：** <https://dreamdojo-world.github.io/>
- **arXiv：** <https://arxiv.org/abs/2602.06949>
- **代码：** <https://github.com/NVIDIA/DreamDojo>（Apache-2.0；2026-09-02 核查）
- **索引来源：** [具身智能研究室 · 42 篇 RL 运动控制长文](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）
- **原始抓取：** [wechat_humanoid_rl_42_survey_2026-05-26.md](../raw/wechat_humanoid_rl_42_survey_2026-05-26.md)（Agent Reach + Camoufox）
- **入库日期：** 2026-05-26
- **一句话说明：** DreamDojo 也是这次新增材料里非常值得单独放大的工作。它做的不是 VLA，而是 robot world model：给定机器人当前观察和动作，预测接下来会发生什么。

## 核心摘录（策展，非全文）

- **在身体系统栈中的位置：** 04 视觉闭环 · 任务接口 · 世界模型，编号 **35/42**。
- **公众号导读要点：** DreamDojo 也是这次新增材料里非常值得单独放大的工作。它做的不是 VLA，而是 robot world model：给定机器人当前观察和动作，预测接下来会发生什么。
- **读者动作：** 方法细节以论文 PDF / 项目页为准；总框架见 [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)。

## 对 wiki 的映射

- [DreamDojo 论文实体](../../wiki/entities/paper-hrl-stack-35-dreamdojo.md) — canonical 详情节点（arXiv:2602.06949）
- [VLA / 世界模型 14 篇阅读路线](../../wiki/overview/vla-wm-reading-roadmap-14-papers-technology-map.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- 姊妹篇 AMP 专题：[wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)

## 一手论文与代码核读（2026-11-17）

- **原文：** [DreamDojo arXiv:2602.06949](https://arxiv.org/html/2602.06949)；[官方仓](https://github.com/NVIDIA/DreamDojo)；[项目](https://dreamdojo-world.github.io/)。
- **核读范围：** 论文潜动作、视频预训练、机器人后训练、蒸馏及长时域滚动、评测、训练代价、局限；核对官方仓 README 与 pretrain/post-train/distillation 文档入口。未运行 GPU 实验。
- **通读：** 已完成

原文把大量第一视角人类视频转换成带连续潜动作条件的世界建模训练信号，再用少量目标本体机器人片段后训练。DreamDojo-HV 约 43,827 小时，混合训练总量约 44,711 小时；“44K 人视频”和“全部训练集”不可混作同一精确口径。蒸馏后的 10 FPS 交互生成属于特定硬件和分辨率实验，不是任意场景通用实时保证。官方发布训练、后训练和 2B/14B 权重，复现仍需大规模算力、可用的数据资产与模型版本对齐。
