# Whole-Body Bilateral Teleoperation with Multi-Stage Object Parameter Estimation for Wheeled Humanoid Locomanipulation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Whole-Body Bilateral Teleoperation with Multi-Stage Object Parameter Estimation for Wheeled Humanoid Locomanipulation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Whole-Body_Bilateral_Teleoperation_with_Multi-Stage_Object_Parameter_Estimation/Whole-Body_Bilateral_Teleoperation_with_Multi-Stage_Object_Parameter_Estimation.html>
- **分类：** 07_Teleoperation
- **arXiv：** <https://arxiv.org/abs/2508.09846>
- **入库日期：** 2026-07-10
- **一句话说明：** 本文提出一个面向轮式人形移动操作的物体感知全身双边遥操作（object-aware whole-body bilateral teleoperation）框架，把遥操作与在线参数估计结合。它顺序整合：① 基于视觉的物体尺寸估计；② 由大型视觉语言模型（VLM）生成的初始参数猜测；③ 一个解耦的分层采样策略（先估质量/质心、再推惯量）。估出的参数用于实时更新机器人的平衡点（equilibrium point）。在搬运约机器人体重 1/3 的负载（抬、送、放）时，框架实现更动态的全身遥操作，同时保持柔顺，并通过物体动力学补偿改善操作跟踪；在自制轮式人形 + 夹爪上实时验证。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-whole-body-bilateral-teleoperation-with-multi-st](../../wiki/entities/paper-notebook-whole-body-bilateral-teleoperation-with-multi-st.md).

## 对 wiki 的映射

- [paper-notebook-whole-body-bilateral-teleoperation-with-multi-st](../../wiki/entities/paper-notebook-whole-body-bilateral-teleoperation-with-multi-st.md)
- 分类父节点：[paper-notebook-category-07-teleoperation](../../wiki/overview/paper-notebook-category-07-teleoperation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Whole-Body_Bilateral_Teleoperation_with_Multi-Stage_Object_Parameter_Estimation/Whole-Body_Bilateral_Teleoperation_with_Multi-Stage_Object_Parameter_Estimation.html>
- 论文：<https://arxiv.org/abs/2508.09846>
