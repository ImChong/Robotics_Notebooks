# Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2511.11218>
- **入库日期：** 2026-07-10
- **一句话说明：** 人形已能与静态场景交互（行走、操作），但动态实时交互仍难。作为迈向快速运动物体交互的一步，本文给出一条 RL 训练流水线，产出人形羽毛球的统一全身控制器，协调步法与击球，且不用动作先验、不用专家示范。训练遵循三阶段课程：① 步法获取；② 精度引导的挥拍生成；③ 任务聚焦精修——使腿与臂共同服务击球目标。部署时用扩展卡尔曼滤波（EKF）估计并预测羽毛球轨迹实现定点击球；并开发一个免预测变体（去掉 EKF 与显式预测）。仿真中双机可连续对打 21 拍；真机出球速度达 19.1 m/s、平均回球落点 4 米；EKF 版与免预测版表现相当。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re](../../wiki/entities/paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md).

## 对 wiki 的映射

- [paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re](../../wiki/entities/paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning.html>
- 论文：<https://arxiv.org/abs/2511.11218>
