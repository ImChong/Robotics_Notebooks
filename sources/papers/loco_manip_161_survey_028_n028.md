# 从专家到通用：走向人形机器人的通用全身控制

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 028/161）

- **标题：** From Experts to a Generalist: Toward General Whole-Body Control for Humanoid Robots
- **类型：** paper
- **Loco-Manip 161 分类：** 01 运控基座与通用全身跟踪
- **机构：** Peking University
- **项目页：** https://beingbeyond.github.io/BumbleBee/
- **发表日期：** 2025年9月2日
- **入库日期：** 2026-06-26
- **一句话说明：** 这篇工作把相机图像/多视角观测、本体状态与关节序列、仿真交互数据转成可跟踪的身体目标，并通过世界模型/视频预测、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把世界模型/视频预测、全身控制器/WBC/MPC放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 01 运控基座与通用全身跟踪，编号 **028/161**。
- **算法实现总结（公众号）：** 这篇工作把相机图像/多视角观测、本体状态与关节序列、仿真交互数据转成可跟踪的身体目标，并通过世界模型/视频预测、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把世界模型/视频预测、全身控制器/WBC/MPC放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 对 wiki 的映射

- [paper-loco-manip-161-028-n028](../../wiki/entities/paper-loco-manip-161-028-n028.md)
- [loco-manip-161-category-01-motion-base-wbt](../../wiki/overview/loco-manip-161-category-01-motion-base-wbt.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
