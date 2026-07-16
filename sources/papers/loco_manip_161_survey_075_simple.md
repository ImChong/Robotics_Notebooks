# SIMPLE

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 075/161）

- **标题：** SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation
- **类型：** paper
- **Loco-Manip 161 分类：** 03 视觉感知驱动的人形移动操作
- **机构：** 南加州大学 Physical Superintelligence (PSI) Lab
- **作者：** Songlin Wei, Zhenhao Ni, Jie Liu, Zhenyu Zhao, Junjie Ye, Hongyi Jing, Junkai Xia, Xiawei Liu, Michael Leong, Liang Heng, Di Huang, Yue Wang
- **arXiv：** 2606.08278
- **项目页：** https://psi-lab.ai/SIMPLE
- **代码：** https://github.com/physical-superintelligence-lab/SIMPLE
- **发表日期：** 2026年6月
- **入库日期：** 2026-06-26（161 策展）；**2026-07-16** 升格 arXiv 深读见 [simple_arxiv_2606_08278.md](simple_arxiv_2606_08278.md)
- **一句话说明：** 人形全身 loco-manipulation **统一仿真 testbed**（MuJoCo 物理 + Isaac Sim 渲染）；60 任务 / 50 场景 / 1000+ 物体；内置运动规划与 VR 遥操作数据采集，并 benchmark VLA/WAM/IL。⚠️ 下方 161 策展「算法实现总结」曾误述为 VLA+世界模型路线，以 arXiv 原文为准。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 03 视觉感知驱动的人形移动操作，编号 **075/161**。
- **算法实现总结（公众号）：** SIMPLE 的实现路径是先把相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据编码成多模态表征，再用VLA 多模态动作模型、世界模型/视频预测预测全身轨迹/动作序列。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。

## 对 wiki 的映射

- [paper-loco-manip-161-075-simple](../../wiki/entities/paper-loco-manip-161-075-simple.md)
- [loco-manip-161-category-03-visuomotor](../../wiki/overview/loco-manip-161-category-03-visuomotor.md)
- 深读归档：[simple_arxiv_2606_08278.md](simple_arxiv_2606_08278.md) · [psi-lab-simple.md](../sites/psi-lab-simple.md) · [simple_usc_psi.md](../repos/simple_usc_psi.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
