# VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation

> 来源归档（ingest · Humanoid Paper Notebooks progress + arXiv 一手资料）

- **标题：** VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation
- **类型：** paper
- **深读状态：** 待撰写（见 [progress.json](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json)）；本仓库已 ingest 策展摘要
- **计划笔记路径：** `papers/04_Loco-Manipulation_and_WBC/VisualMimic__Visual_Humanoid_Loco-Manipulation_via_Motion_Tracking_and_Generatio/VisualMimic__Visual_Humanoid_Loco-Manipulation_via_Motion_Tracking_and_Generatio.md`
- **分类：** 04_Loco-Manipulation_and_WBC
- **路线：** Loco-Manipulation / Visual Sim2Real
- **arXiv：** <https://arxiv.org/abs/2509.20322>
- **项目页：** <https://visualmimic.github.io/>
- **代码：** <https://github.com/visualmimic/VisualMimic>
- **入库日期：** 2026-06-12（progress 锚点 2026-06-11；2026-06-12 补充一手资料）
- **一句话说明：** **视觉分层 sim2real**：人类动作蒸馏的 **关键点低层跟踪器** + **深度 visuomotor 高层生成器**，真机零样本完成多样全身 loco-manipulation 并泛化户外。

## 核心摘录（策展，非全文）

- **分层结构：** 低层 $\pi_{\mathrm{tracker}}$ 任务无关、共享；高层 $\pi_{\mathrm{generator}}$ 每任务训练；接口为 **root + 5 关键点**（头/双手/双足）。
- **双 teacher–student：** 低层 motion→keypoint（DAgger）；高层 state→depth vision（特权物体状态教师 → 视觉学生）。
- **稳定技巧：** 低层训时加噪；高层动作 clip 到 **HMS**；仿真深度 **heavy masking** 抗 visual gap。
- **数据：** GMR 重定向 AMASS + OMOMO；**无需配对人–物交互 MoCap**。
- **真机亮点：** 0.5 kg 抬至 1 m；3.8 kg 大箱全身 push；足球盘带与双脚踢球；户外鲁棒。
- **一手资料主归档：** [visualmimic_arxiv_2509_20322.md](./visualmimic_arxiv_2509_20322.md)；仓库 [visualmimic.md](../repos/visualmimic.md)；项目页 [visualmimic-github-io.md](../sites/visualmimic-github-io.md)。

## 对 wiki 的映射

- [paper-notebook-visualmimic](../../wiki/entities/paper-notebook-visualmimic.md)
- [loco-manipulation](../../wiki/tasks/loco-manipulation.md)
- [paper-twist](../../wiki/entities/paper-twist.md)
- [videomimic](../../wiki/entities/videomimic.md)
- [paper-resmimic](../../wiki/entities/paper-resmimic.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2509.20322>
- 项目页：<https://visualmimic.github.io/>
- 代码：<https://github.com/visualmimic/VisualMimic>
- [Humanoid Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json)
