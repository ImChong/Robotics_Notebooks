# Control Operators for Interactive Character Animation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Control Operators for Interactive Character Animation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Control_Operators_for_Interactive_Character_Animation/Control_Operators_for_Interactive_Character_Animation.html>
- **分类：** 14_Human_Motion
- **入库日期：** 2026-06-07
- **一句话说明：** 把「控制输入 → 神经网络」这件原本需要 ML 专家手工设计的事，拆解成一组有语义、可组合的「控制算子（Control Operator）」：每个算子对设计师来说是一个直观概念（"沿这条轨迹走""朝这个目标看""按摇杆方向/速度移动""在某时刻到达某位置"），对网络来说则对应一段固定的编码结构。把若干算子拼起来，非技术用户就能自己训练出带多技能、多控制模式的学习型角色控制器——本文在 Learned Motion Matching 变体 和一个新的流匹配（flow-matching）自回归模型上都做了演示，并通过工业界从业者的用户研究验证其易用性。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-control-operators-for-interactive-character-anim](../../wiki/entities/paper-notebook-control-operators-for-interactive-character-anim.md).

## 对 wiki 的映射

- [paper-notebook-control-operators-for-interactive-character-anim](../../wiki/entities/paper-notebook-control-operators-for-interactive-character-anim.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Control_Operators_for_Interactive_Character_Animation/Control_Operators_for_Interactive_Character_Animation.html>

