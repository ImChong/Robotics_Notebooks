# Pro-HOI

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 074/161）

- **标题：** Pro-HOI: Perceptive Root-guided Humanoid-Object Interaction
- **类型：** paper
- **Loco-Manip 161 分类：** 03 视觉感知驱动的人形移动操作
- **机构：** Institute of Artificial Intelligence (TeleAI), China Telecom、Zhejiang University、University of Science and Technology of China、ShanghaiTech University
- **项目页：** https://pro-hoi.github.io/
- **发表日期：** 2026年3月1日
- **入库日期：** 2026-06-26
- **一句话说明：** Pro-HOI 先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用扩散策略/流匹配、IK/动作重定向、全身控制器/WBC/MPC生成全身轨迹/动作序列。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 03 视觉感知驱动的人形移动操作，编号 **074/161**。
- **算法实现总结（公众号）：** Pro-HOI 先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用扩散策略/流匹配、IK/动作重定向、全身控制器/WBC/MPC生成全身轨迹/动作序列。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 对 wiki 的映射

- [paper-loco-manip-161-074-pro-hoi](../../wiki/entities/paper-loco-manip-161-074-pro-hoi.md)
- [loco-manip-161-category-03-visuomotor](../../wiki/overview/loco-manip-161-category-03-visuomotor.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)

## 项目页与开源状态核查（2026-07-22）

- **论文：** arXiv:2603.01126；项目页 <https://pro-hoi.github.io/>。
- **代码：** 未确认官方可运行仓库。
- **关键数字：** OOD MuJoCo Ours w/ FR. grasp success 99.93%、task success 88.38%；真机中速 21/28 grasp success；系统报告超过 15 个连续搬运循环。
- **wiki 深化：** [paper-loco-manip-161-074-pro-hoi](../../wiki/entities/paper-loco-manip-161-074-pro-hoi.md)。
