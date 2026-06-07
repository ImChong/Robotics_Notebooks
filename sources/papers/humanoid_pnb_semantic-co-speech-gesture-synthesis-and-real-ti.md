# Semantic Co-Speech Gesture Synthesis and Real-Time Control for Humanoid Robots

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Semantic Co-Speech Gesture Synthesis and Real-Time Control for Humanoid Robots
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Semantic_Co-Speech_Gesture_Synthesis_and_Real-Time_Control_for_Humanoid_Robots/Semantic_Co-Speech_Gesture_Synthesis_and_Real-Time_Control_for_Humanoid_Robots.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2512.17183>
- **入库日期：** 2026-06-07
- **一句话说明：** 论文把"机器人讲话的同时做出语义对齐的手势"这件事拆成 语义检索 + 自回归生成 + 人到机重定向 + 全身跟踪 四段流水线：用 LLM 从语料库里检索与语义高度相关的人体手势片段、用 Motion-GPT 自回归补全长时间序列、用 General Motion Retargeting (GMR) 把人体动作迁到 Unitree G1 上，最后用强化学习训出的 MotionTracker 把这套带有语义的参考动作在真机上稳定、实时地跟出来。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti](../../wiki/entities/paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti.md).

## 对 wiki 的映射

- [paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti](../../wiki/entities/paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Semantic_Co-Speech_Gesture_Synthesis_and_Real-Time_Control_for_Humanoid_Robots/Semantic_Co-Speech_Gesture_Synthesis_and_Real-Time_Control_for_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2512.17183>
