# NaVILA: Legged Robot Vision-Language-Action Model for Navigation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** NaVILA: Legged Robot Vision-Language-Action Model for Navigation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/NaVILA_Legged_Robot_Vision-Language-Action_Model_for_Navigation/NaVILA_Legged_Robot_Vision-Language-Action_Model_for_Navigation.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2412.04453>
- **项目页：** <https://navila-bot.github.io/>
- **代码：** <https://github.com/AnjieCheng/NaVILA>（**已开源**，Apache-2.0；训练 / 评测 / 权重 / 标注已发布）
- **入库日期：** 2026-06-07
- **一句话说明：** NaVILA 提出了一个两层分层框架，将高层视觉 - 语言 - 动作（VLA）推理出的自然语言指令转换为低层足式运动控制，实现了人类自然语言指令到复杂地面导航的高效映射。

## 核心摘录（策展，非全文）

- **两层接口：** VLA 输出“forward 75 cm / turn 30°”参数化语言动作；视觉 RL locomotion policy 转换为关节控制。
- **数据：** R2R / RxR / EnvDrop / ScanQA / VQA + 2k YouTube touring videos→20k trajectories；原视频仅给 IDs 与 annotations。
- **结果：** R2R-CE SR 54%；VLN-CE-Isaac Go2/H1 Vision SR 50.2/45.3%；真机 Go2 / Booster T1 共 25 instructions × 3 repeats。
- **开源边界：** core repo、HF 权重 / 标注、NaVILA-Bench 和 legged-loco 均有入口；旧 Habitat 0.1.7 与多仓库组合仍是主要复现成本。
- 知识归纳见 wiki 实体页：[paper-notebook-navila-legged-robot-vision-language-action-model](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md).

## 对 wiki 的映射

- [paper-notebook-navila-legged-robot-vision-language-action-model](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)
- [NaVILA 项目页归档](../sites/navila.md)
- [NaVILA 仓库归档](../repos/navila.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/NaVILA_Legged_Robot_Vision-Language-Action_Model_for_Navigation/NaVILA_Legged_Robot_Vision-Language-Action_Model_for_Navigation.html>
- 论文：<https://arxiv.org/abs/2412.04453>
