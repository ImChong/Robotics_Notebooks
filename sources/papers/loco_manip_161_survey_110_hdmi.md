# HDMI

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 110/161）

- **标题：** HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos
- **类型：** paper
- **Loco-Manip 161 分类：** 05 动捕、人类视频与交互动作规划
- **机构：** （见原文）
- **项目页：** https://hdmi-humanoid.github.io
- **arXiv：** https://arxiv.org/abs/2509.16757
- **代码：** https://github.com/LeCAR-Lab/HDMI
- **发表日期：** 2025年9月27日
- **入库日期：** 2026-06-26
- **复核日期：** 2026-07-22
- **一句话说明：** HDMI 先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用IK/动作重定向、分层技能/专家策略生成全身轨迹/动作序列。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 05 动捕、人类视频与交互动作规划，编号 **110/161**。
- **算法实现总结（公众号）：** HDMI 先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用IK/动作重定向、分层技能/专家策略生成全身轨迹/动作序列。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。
- **论文复核修正：** arXiv / 项目页显示 HDMI 的主线不是高层技能路由，而是 **monocular RGB video → robot/object/contact reference → robot-object co-tracking RL**；三项关键设计为 unified object representation、residual action space、unified interaction reward。
- **结果与开源：** Unitree G1 真机 67 次连续 door traversal、6 类真机任务、14 类仿真任务；官方代码已开源于 [LeCAR-Lab/HDMI](https://github.com/LeCAR-Lab/HDMI)。

## 对 wiki 的映射

- [paper-hrl-stack-06-hdmi](../../wiki/entities/paper-hrl-stack-06-hdmi.md)
- [loco-manip-161-category-05-mocap-human-video](../../wiki/overview/loco-manip-161-category-05-mocap-human-video.md)
- [loco-manip-contact-category-01-contact-data](../../wiki/overview/loco-manip-contact-category-01-contact-data.md)
- [hdmi-project](../sites/hdmi-project.md)
- [hdmi repo](../repos/hdmi.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 接触横切面编译：[wechat_embodied_ai_lab_loco_manip_contact_survey.md](../blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
- 项目页：<https://hdmi-humanoid.github.io>
- arXiv：<https://arxiv.org/abs/2509.16757>
- 代码：<https://github.com/LeCAR-Lab/HDMI>
