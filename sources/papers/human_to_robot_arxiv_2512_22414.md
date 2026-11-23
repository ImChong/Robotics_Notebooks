# Emergence of Human to Robot Transfer in Vision-Language-Action Models（arXiv:2512.22414）

> 来源归档（ingest）

- **标题：** Emergence of Human to Robot Transfer in Vision-Language-Action Models
- **短名：** PI human-to-robot transfer
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2512.22414>
- **PI 官方论文 PDF：** <https://www.pi.website/download/human_to_robot.pdf>（13 页；用户提供）
- **会议：** RSS 2026，<https://www.roboticsproceedings.org/rss22/p072.html>
- **项目页：** <https://www.pi.website/research/human_to_robot>
- **机构：** 物理智能（Physical Intelligence）；佐治亚理工学院（Georgia Tech）
- **入库日期：** 2026-09-28
- **核对范围：** 2026-11-23 直接核对 PI 官方 13 页 PDF 的 Sec. I–VI、参考文献、Appendix A–C、Fig. 1–13，以及 PI 官方研究博客；此前 2026-11-22 核对了 arXiv HTML v1。官方研究页仅列 Paper/PDF 入口，未列本文独立公开的训练代码、人视频或机器人数据集；未在本地复现实验。
- **通读：** 已完成
- **一句话说明：** π₀.₅ 预训练足够多样之后，把第一人称人视频当成另一种本体一起微调，人→机器人迁移才会出现。

## 开源状态（步骤 2.5，2026-09-28）

- **确认未开源**：项目页、arXiv 与 RSS 页未列训练代码、人视频或机器人数据。引用里的第三方仓库不是本文实现。

## 核心摘录（面向 wiki 编译）

- 动作用 3D 手部位置，不做人机外观对齐、不把人手生成成夹爪。人视频只在微调出现，预训练仍是机器人数据。
- 博客称四个只出现在人视频里的泛化设置上，加入人数据后性能大约 2 倍。论文写迁移幅度随预训练多样性上升：多样性 0% / 25% 时共训人数据几乎无收益，75% / 100% 以及跨本体混合后收益变大。
- 论文还写：Sort Eggs 与 Dresser 上，人数据微调接近目标机器人域内数据；Bussing 上目标机器人数据仍然更有效。
- **对 wiki 的映射：** [paper-pi-human-to-robot](../../wiki/entities/paper-pi-human-to-robot.md)

## 官方 PDF 补充核验（2026-11-23）

- **数据采集（Sec. IV-A）：** Bussing、Spice、Dresser、Sort Eggs 各 3/3/3/5 小时；头戴相机和可选双腕相机，经视觉 SLAM 估计头部 6D 轨迹，追踪每只手 17 个三维关键点；以手掌/中指/无名指计算虚拟末端 6-DoF 相对位姿。
- **人机动作接口不是逐维相同：** 机器人数据为双末端 6D+夹爪及 2D 底盘动作（16 维），人视频为双手 6D 相对末端加 6D 头部运动（18 维），不估计人手夹爪开合。所谓无显式对齐指不额外设计人机外观/分布对齐损失；仍需结构化手部运动提取与语义近似对齐。
- **训练配方（Sec. III–IV）：** 不同多样性的机器人数据预训练 π₀.₅；微调 50:50 混合人视频与相近机器人任务示范，训练子任务文本预测、离散 FAST action tokens、连续 flow-matching 动作预测。
- **四项评测（Appendix A）：** Bussing 为 9 件物品正确归置的归一化分数；Spice 为 3 瓶全部放回的二值成功；Dresser 为项链、发夹正确入对应收纳盒的二值成功；Sort Eggs 为 12 颗鸡蛋按颜色入盒并奖励关盒动作的归一化分数。每项评测 20–40 次，误差棒为 1 标准误，跨任务分数不能全部当成同一种成功率。
- **结果及证据界限（Sec. V、Appendix B–C）：** Spice 0.32→0.71、Dresser 0.25→0.50、Bussing 0.53→0.63；鸡蛋颜色分拣准确率 57%→78%。不同预训练档位的比例指实验数据覆盖多样性，不代表可推广的模型参数量阈值；t-SNE 特征重叠仅提供相关性观察。
- **对 wiki 的映射：** [独立论文详情](../../wiki/entities/paper-pi-human-to-robot.md)。
