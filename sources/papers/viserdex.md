# ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation / 鲁棒灵巧手内重定向的视觉仿真到真实迁移

- **论文：** <https://arxiv.org/abs/2604.11138>；[PDF v1](https://arxiv.org/pdf/2604.11138)
- **作者：** Arjun Bhardwaj, Maximum Wilder-Smith, Mayank Mittal, Vaishakh Patil, Marco Hutter（作者排序以原文为准）
- **机构：** 苏黎世联邦理工学院（ETH Zurich）；NVIDIA
- **会议：** RSS 2026
- **核对：** 2026-11-03，论文 v1 正文、方法、实验、局限及附录。
- **项目页：** https://rffr.leggedrobotics.com/works/viserdex/
- **对 wiki 的映射：** [viserdex](../../wiki/entities/paper-viserdex.md)

## 方法与实验摘录

- 在高斯参数空间做聚类空间、颜色和全局扰动，生成多样外观；图像训练关键点回归器，再通过刚体配准恢复姿态。
- 教师用性能驱动课程学习，学生用在线 DAgger 蒸馏，注入延迟、噪声和跟踪故障；估计姿态是学生输入。
- 论文面向不同物体采用各自策略；公开仓库说明释放的是高斯渲染、姿态估计训练评测和生成轨迹用的 Isaac Lab 默认任务，不能视作完整论文真机部署栈。

- 每个物体、每种光照运行 5 次，成功阈值为姿态误差 0.4 rad。五物体正常光照平均连续成功 37.6，强干扰光照 25.4；各物体差异很大，强干扰下药瓶 4.2、地球仪 76.2。
- 同一立方体上正常光照 35.4 次，对照 DeXtreme 27.8 次。替换为 FoundationPose 的部署比较约 0.4 次，论文指出频率和跟踪丢失因素。
- 与 GoalDex 的 80/100 单目标成功率不是同一指标或实验协议。

## 局限

视觉遮挡和接触力不可直接观测；论文指出物体实例泛化仍需研究。药瓶标签摩擦未建模导致性能下降。公开数据集 [ViserDexSplats](https://huggingface.co/datasets/leggedrobotics/ViserDexSplats) 为 CC BY-NC 4.0，代码仓为 BSD-3-Clause。
