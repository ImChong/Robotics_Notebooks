# Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Go_to_Zero__Towards_Zero-shot_Motion_Generation_with_Million-scale_Data/Go_to_Zero__Towards_Zero-shot_Motion_Generation_with_Million-scale_Data.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2507.07095>
- **入库日期：** 2026-07-10
- **一句话说明：** 本文要把文本到动作（text-to-motion）推向零样本泛化的新阶段。为此提出一套高效动作标注机制：从网络规模人类动作视频中自主采集人类动作——用运动学回归（kinematic regression）从无标注视频中提取动作，并用先进视觉语言模型生成语义丰富的描述（caption）。据此构建 MotionMillion ——迄今最大的人类动作数据集（2000+ 小时、200 万条高质量动作序列）。还提出 MotionMillion-Eval ——最全面的零样本动作生成评测基准。作者把模型规模化到 7B 参数并在 MotionMillion-Eval 上验证：结果对域外（out-of-domain）与复杂组合（compositional）动作展现强泛化，是迈向零样本人类动作生成的重要一步。代码开源。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-go-to-zero-towards-zero-shot-motion-generation-w](../../wiki/entities/paper-notebook-go-to-zero-towards-zero-shot-motion-generation-w.md).

## 对 wiki 的映射

- [paper-notebook-go-to-zero-towards-zero-shot-motion-generation-w](../../wiki/entities/paper-notebook-go-to-zero-towards-zero-shot-motion-generation-w.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Go_to_Zero__Towards_Zero-shot_Motion_Generation_with_Million-scale_Data/Go_to_Zero__Towards_Zero-shot_Motion_Generation_with_Million-scale_Data.html>
- 论文：<https://arxiv.org/abs/2507.07095>
