# OpenHLM

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 154/161）

- **标题：** OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation
- **类型：** paper
- **Loco-Manip 161 分类：** 09 人形 VLA、世界模型与通用操作
- **机构：** Tsinghua University、Shanghai Qi Zhi Institute、Spirit AI
- **项目页：** https://openhlm-project.github.io/
- **发表日期：** 2026年6月20日
- **入库日期：** 2026-06-26
- **一句话说明：** OpenHLM 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、扩散策略/流匹配、VLA 多模态动作模型预测全身轨迹/动作序列、低层控制器目标、地形/场景表征。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 09 人形 VLA、世界模型与通用操作，编号 **154/161**。
- **算法实现总结（公众号）：** OpenHLM 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、扩散策略/流匹配、VLA 多模态动作模型预测全身轨迹/动作序列、低层控制器目标、地形/场景表征。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 对 wiki 的映射

- [paper-loco-manip-161-154-openhlm](../../wiki/entities/paper-loco-manip-161-154-openhlm.md)
- [loco-manip-161-category-09-vla-world-models](../../wiki/overview/loco-manip-161-category-09-vla-world-models.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)

## 项目页与开源状态核查（2026-07-22）

- **论文/项目：** arXiv:2606.22174；项目页 <https://openhlm-project.github.io/>。
- **代码：** 项目页未列官方 GitHub/可运行代码。
- **关键数字：** 水果长程基准 OpenHLM 1.14 h demos 达 87.5% task progress；GR00T N1.6 2.70 h 为 57.5%，Ψ0 为 48.8%；12 个语言任务平均 task progress 超 90%。
- **wiki 深化：** [paper-loco-manip-161-154-openhlm](../../wiki/entities/paper-loco-manip-161-154-openhlm.md)。
