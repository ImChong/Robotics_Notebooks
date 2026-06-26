# 具有实时基础地形重建的步态自适应感知人形运动

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 142/161）

- **标题：** Gait-Adaptive Perceptive Humanoid Locomotion with Real-Time Under-Base Terrain Reconstruction
- **类型：** paper
- **Loco-Manip 161 分类：** 08 硬件平台、感知配置与部署扩展
- **机构：** （见原文）
- **项目页：** https://ga-phl.github.io/
- **发表日期：** 2025年12月8日
- **入库日期：** 2026-06-26
- **一句话说明：** 这篇工作先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用下视深度相机和 U-Net 高度图重建、步态相位/频率调节、教师-学生知识迁移生成关节位置/力矩命令、地形/场景表征。关键点是把地形重建、步态相位和全身姿态放进同一个控制回路，而不是把感知和运控拆成松散串联。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 08 硬件平台、感知配置与部署扩展，编号 **142/161**。
- **算法实现总结（公众号）：** 这篇工作先从相机图像/多视角观测、本体状态与关节序列、深度/点云/高度图恢复场景、目标或运动表征，再用下视深度相机和 U-Net 高度图重建、步态相位/频率调节、教师-学生知识迁移生成关节位置/力矩命令、地形/场景表征。关键点是把地形重建、步态相位和全身姿态放进同一个控制回路，而不是把感知和运控拆成松散串联。

## 对 wiki 的映射

- [paper-loco-manip-161-142-n142](../../wiki/entities/paper-loco-manip-161-142-n142.md)
- [loco-manip-161-category-08-hardware-deployment](../../wiki/overview/loco-manip-161-category-08-hardware-deployment.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
