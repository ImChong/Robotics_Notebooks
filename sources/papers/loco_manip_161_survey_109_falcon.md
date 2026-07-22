# FALCON

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 109/161）

- **标题：** FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation
- **类型：** paper
- **Loco-Manip 161 分类：** 05 动捕、人类视频与交互动作规划
- **机构：** Carnegie Mellon University
- **项目页：** https://lecar-lab.github.io/falcon-humanoid
- **发表日期：** 2025年11月16日
- **入库日期：** 2026-06-26
- **一句话说明：** FALCON 主要解决数据闭环：用本体状态与关节序列、人类视频/动捕轨迹、遥操作/外骨骼数据采集人类操作和机器人状态，再通过全身控制器/WBC/MPC转成可训练、可复用的关节位置/力矩命令、末端执行器/腕手目标。关键点是把全身控制器/WBC/MPC放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 05 动捕、人类视频与交互动作规划，编号 **109/161**。
- **算法实现总结（公众号）：** FALCON 主要解决数据闭环：用本体状态与关节序列、人类视频/动捕轨迹、遥操作/外骨骼数据采集人类操作和机器人状态，再通过全身控制器/WBC/MPC转成可训练、可复用的关节位置/力矩命令、末端执行器/腕手目标。关键点是把全身控制器/WBC/MPC放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 对 wiki 的映射

- [paper-loco-manip-161-109-falcon](../../wiki/entities/paper-loco-manip-161-109-falcon.md)
- [loco-manip-161-category-05-mocap-human-video](../../wiki/overview/loco-manip-161-category-05-mocap-human-video.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)

## 项目页与开源状态核查（2026-07-22）

- **论文/项目：** arXiv:2505.06776；项目页 <https://lecar-lab.github.io/falcon-humanoid>。
- **代码：** <https://github.com/LeCAR-Lab/FALCON>，MIT；已发布 training、sim2sim、sim2real。
- **关键数字：** payload transport 0-20 N，cart pulling 0-100 N，door opening 0-40 N；upper-body joint tracking 精度约 2× baseline。
- **wiki 深化：** [paper-loco-manip-161-109-falcon](../../wiki/entities/paper-loco-manip-161-109-falcon.md) 已补源码运行时序图。
