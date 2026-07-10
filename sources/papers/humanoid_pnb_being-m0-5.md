# Being-M0.5: A Real-Time Controllable Vision-Language-Motion Model

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Being-M0.5: A Real-Time Controllable Vision-Language-Motion Model
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Being-M0.5__A_Real-Time_Controllable_Vision-Language-Motion_Model/Being-M0.5__A_Real-Time_Controllable_Vision-Language-Motion_Model.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2508.07863>
- **入库日期：** 2026-07-10
- **一句话说明：** 人类动作生成潜力巨大，但现有视觉-语言-动作模型（VLMM）实用部署受限。作者指出可控性是主瓶颈，体现在五方面：对多样人类指令响应不足、姿态初始化能力有限、长序列表现差、对未见场景处理不足、缺乏对各身体部位的细粒度控制。为此提出Being-M0.5，并引入 HuMo100M ——迄今最大最全的人类动作数据集（500 万+ 自采动作序列、1 亿条多任务指令实例、细粒度部位级标注）。方法用部位感知残差量化（part-aware residual quantization）做动作 token 化，实现逐部位的精细控制。模型在多个动作生成基准上达 SOTA，同时保持实时执行效率。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-being-m0-5-a-real-time-controllable-vision-langu](../../wiki/entities/paper-notebook-being-m0-5-a-real-time-controllable-vision-langu.md).

## 对 wiki 的映射

- [paper-notebook-being-m0-5-a-real-time-controllable-vision-langu](../../wiki/entities/paper-notebook-being-m0-5-a-real-time-controllable-vision-langu.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Being-M0.5__A_Real-Time_Controllable_Vision-Language-Motion_Model/Being-M0.5__A_Real-Time_Controllable_Vision-Language-Motion_Model.html>
- 论文：<https://arxiv.org/abs/2508.07863>
