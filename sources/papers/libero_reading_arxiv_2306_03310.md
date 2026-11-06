# LIBERO 论文通读（终身机器人操作学习基准视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/libero-benchmark.md`）

- **标题：** LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning
- **作者：** Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone
- **类型：** paper / benchmark / lifelong-imitation-learning
- **原文：** <https://arxiv.org/abs/2306.03310>（2026-11 通过 arXiv PDF 获取全文，pdftotext 提取）
- **一句话说明：** 提出可程序化生成任务的终身机器人操作学习基准 LIBERO，给出 130 个任务、人类遥操作演示，并系统比较三种策略架构与三种终身学习算法。
- **核对范围：** 通读正文 §1–§7（含 §4 基准构成与 §5 全部实验 Q1–Q6、结论与局限）；附录 A（网络结构）、B.1（算法）、C.1（任务套件）、D（实验设置）与 E 的主要表格 / 说明已阅读，附录 E.4 的注意力可视化图只读文字结论，未逐图核对；未运行官方代码、未复现数值。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题设定（§2–§3）：** 终身决策学习（LLDM）用 MDP 形式化；区分陈述性知识（空间关系、物体）与程序性知识（动作、行为）的迁移，并提出五个研究主题：分布偏移类型、网络结构、终身算法、任务顺序鲁棒性、预训练。
- **程序化任务生成（§4.1）：** 从 Ego4D 语言标注提取行为模板生成指令；初始状态分布 µ0 与目标用 PDDL 描述，目标为谓词合取（如 On、In、Open、TurnOff）；底层基于 robosuite，所有谓词为真时仿真终止。
- **任务套件（§4.2）：** LIBERO-Spatial / Object / Goal 各 10 个任务，分别受控地考察空间关系、物体类别、任务目标的迁移；LIBERO-100 含 100 个纠缠迁移任务，拆为 90 个短程任务（LIBERO-90，用于预训练）与 10 个长程任务（论文称 LIBERO-Long，即页面中的 LIBERO-10）。
- **算法（§4.3）：** ER（记忆）、EWC（正则）、PackNet（动态结构），外加顺序微调 SeqL（下界）和多任务学习 MTL（上界）。
- **架构（§4.4）：** ResNet-RNN、ResNet-T、ViT-T；语言用预训练 BERT 嵌入，GMM 输出头；统一用行为克隆，每任务 50 条 SpaceMouse 遥操作演示。
- **指标（§5.1，附录 D）：** FWT、NBT、AUC，均基于成功率；每 5 epoch 评测一次，每次 20 条 rollout、最长 600 步，训练 50 epoch，3 个种子。
- **架构结论（Q1–Q2，表 1）：** Transformer 时间骨干整体优于 RNN；ER 下 ResNet-T 在除 Object 外的套件更好，ViT 在物体种类多时有优势；PackNet 下两者在多数套件无显著差异，Long 上 ViT-T 更好。
- **算法结论（Q1、Q3，表 2）：** SeqL 在所有套件的 FWT 最高，说明所测终身算法都损害前向迁移；PackNet 在 Spatial/Object/Goal 上最佳但在 Long 上被 ER 超过（作者推测容量被切分）；EWC 比 SeqL 还差；ER 在各套件稳健。
- **语言嵌入（Q4，表 3）：** BERT / CLIP / GPT-2 / Task-ID 嵌入在 Long 上无统计显著差异，作者解释为句向量近似词袋，仅起区分任务作用。
- **任务顺序（Q5，图 4）：** 不同顺序下性能差异明显，对 PackNet 统计显著。
- **预训练（Q6，图 5）：** 在 LIBERO-90 上做朴素监督预训练会损害下游终身学习，需更好的预训练技术。
- **结论与局限（§7）：** 未来方向为更好的空间 / 时间处理结构、改善前向迁移的算法、有效预训练；仅为初步研究；仿真环境、行为克隆单一范式、成功率为主指标；社会影响方面提到长期需关注用户隐私。

## 对 wiki 的映射

- [`wiki/entities/libero-benchmark.md`](../../wiki/entities/libero-benchmark.md)
