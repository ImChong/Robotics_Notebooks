# Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记 + 项目页开源核查）

- **标题：** Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning/Humanoid_Whole-Body_Badminton_via_Multi-Stage_Reinforcement_Learning.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2511.11218>
- **项目页：** <https://humanoid-badminton.github.io/Humanoid-Whole-Body-Badminton-via-Multi-Stage-Reinforcement-Learning/>
- **入库日期：** 2026-07-10
- **再核日期：** 2026-07-28
- **一句话说明：** 无先验三阶段全身羽毛球 RL；仿真 21 连拍；真机出球最高 19.1 m/s；代码待发布。

## 核心摘录（策展，非全文）

- S1 步法 → S2 精度引导挥拍（σ 收紧）→ S3 去掉接近/步态塑形做任务精修。
- 部署：EKF 目标已知 vs 免预测（当前球 + 5 帧历史）。
- 平台：Phybot C1 1.28 m / 21 DoF；MoCap 基座 + 球尖。
- **开源核查（2026-07-28）：** 组织仓声明 code will be released soon；**无可运行入口**。

## 对 wiki 的映射

- [paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re](../../wiki/entities/paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md)
- [paper-notebook-learning-human-like-badminton-skills-for-humanoi](../../wiki/entities/paper-notebook-learning-human-like-badminton-skills-for-humanoi.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：见上
- 项目页：[humanoid-badminton-multi-stage-rl.md](../sites/humanoid-badminton-multi-stage-rl.md)
- 论文：<https://arxiv.org/abs/2511.11218>
