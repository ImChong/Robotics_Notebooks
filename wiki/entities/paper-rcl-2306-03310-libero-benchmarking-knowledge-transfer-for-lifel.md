---
type: entity
tags: [paper, robot-manipulation, lifelong-learning, benchmark, imitation-learning, dataset, simulation]
status: complete
updated: 2026-10-03
arxiv: "2306.03310"
venue: "NeurIPS 2023 Datasets and Benchmarks Track"
code: https://github.com/Lifelong-Robot-Learning/LIBERO
summary: "LIBERO 提供 130 个语言条件机器人操作任务与人类遥操作演示，将终身学习中的知识迁移拆成空间关系、物体、目标及其组合变化，并系统比较策略架构与持续学习算法。"
related:
  - ../entities/libero-benchmark.md
  - ../tasks/manipulation.md
  - ../entities/paper-actfovea.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2306_03310_libero-benchmarking-knowledge-transfer-f.md
  - ../../sources/repos/libero-benchmark.md
---

# LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning

**LIBERO** 是面向机器人操作的终身学习基准。论文把知识迁移问题设计成一组可控的任务变化，并提供基准任务、仿真环境和人类遥操作演示数据，让研究者比较策略能否把已学到的知识迁移到后续任务。

- **作者：** Bo Liu、Yifeng Zhu、Chongkai Gao、Yihao Feng、Qiang Liu、Yuke Zhu、Peter Stone
- **发表：** NeurIPS 2023 Datasets and Benchmarks Track
- **论文：** [NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html) · [PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf) · [arXiv:2306.03310](https://arxiv.org/abs/2306.03310)
- **代码：** [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- **资料：** [项目主页](https://libero-project.github.io/) · [文档](https://lifelong-robot-learning.github.io/LIBERO/) · [官方数据页](https://libero-project.github.io/datasets) · [Hugging Face 数据集](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)

## 一句话理解

把机器人操作任务按“物体变了、空间关系变了、目标变了、这些因素一起变了”分成几类，再观察机器人在连续学习这些任务时，能否把旧知识用到新任务上，同时保留旧任务能力。

## 为什么重要

单任务成功率很难说明机器人是否真正学会迁移。连续学新任务时，策略可能忘掉旧技能；也可能因为物体、摆放或目标发生变化而无法复用已有知识。LIBERO 用标准化任务套件和演示数据，把这些变化分别控制起来，便于比较方法及分析失败原因。

## 基准组成

论文提出四个任务套件，共 **130 个语言条件操作任务**：

| 套件 | 任务数 | 主要考察的变化 |
|---|---:|---|
| LIBERO-Spatial | 10 | 物体之间的空间关系和摆放布局 |
| LIBERO-Object | 10 | 操作对象类别 |
| LIBERO-Goal | 10 | 任务目标 |
| LIBERO-100 | 100 | 物体、布局与目标等知识的组合迁移 |

LIBERO-100 在基准设置中进一步划分为 **LIBERO-90**（用于预训练）与 **LIBERO-10**（用于下游终身学习评测）。官方数据包括人类遥操作演示；项目说明还列出工作区与腕部 RGB 图像、本体状态、语言任务描述和 PDDL 场景描述等内容。

## 方法与评测设置

论文使用行为克隆（Behavioral Cloning, BC）从演示轨迹学习操作策略，以便在有限计算资源下比较终身学习设定。它研究三种视觉-运动策略架构：

- **ResNet-RNN：** ResNet 编码视觉输入，LSTM 汇总时间信息。
- **ResNet-T：** ResNet 视觉特征与 Transformer 时间骨干结合。
- **ViT-T：** Vision Transformer 处理视觉输入，并以 Transformer 建模时间序列。

比较的学习方案包括顺序微调和多任务学习基线，以及 Experience Replay（ER）、Elastic Weight Consolidation（EWC）和 PackNet 等终身学习方法。论文主要使用任务成功率评估，并研究任务顺序、策略结构、算法选择和预训练对迁移的影响。

## 论文报告的主要发现

1. **架构和算法都影响迁移。** Transformer 时间骨干在抽象时序信息方面表现突出；不同视觉编码器在不同类型的知识迁移上各有强项，没有一种架构对所有套件都最好。
2. **防遗忘不等于更强的前向迁移。** 在论文比较的设定中，ER、EWC、PackNet 等方法能缓解遗忘，但总体上顺序微调的前向迁移表现更好。
3. **任务语言嵌入未必带来提升。** 使用语义丰富的任务描述嵌入，表现并未优于使用任务 ID 嵌入。
4. **朴素监督预训练可能适得其反。** 在大规模离线数据上直接做监督预训练，可能降低后续终身学习表现。

以上结论对应论文的任务、策略和训练协议；复现或横向比较时，应以原文实验设置为准。

## 如何使用

1. 从官方仓库安装环境，查看任务套件、策略配置和评估脚本。
2. 使用官方脚本下载对应套件的遥操作演示数据；README 说明可选择 Hugging Face 下载来源。
3. 选定 suite、策略和终身学习算法，按统一任务顺序及成功率协议评测。
4. 对比 LIBERO-Spatial、Object、Goal 与 LIBERO-90/10 的结果，定位变化来自布局、物体、目标还是它们的组合。

本页不复述完整安装步骤，依赖版本、命令和数据文件结构以[官方 README](https://github.com/Lifelong-Robot-Learning/LIBERO#readme)及[文档](https://lifelong-robot-learning.github.io/LIBERO/)为准。

## 适用范围与限制

- LIBERO 是仿真中的机器人操作基准，适合研究终身模仿学习、知识迁移、任务顺序和策略结构。
- 在 LIBERO 上的结果不能直接等同于真实机械臂上的性能或 sim-to-real 能力。
- 不同 LIBERO 扩展版、任务子集、训练数据和评估协议可能不同；比较分数前先核对具体套件、初始状态、rollout 数和训练设置。
- 本文是 2023 年提出的基准工作。后续如使用更新的仓库版本或扩展套件，应注明版本，避免将新增设置归到原论文。

## 关联页面

- [LIBERO 基准与工程入口](./libero-benchmark.md)
- [机器人操作](../tasks/manipulation.md)
- [ActFovea：LIBERO 上的 VLA 扰动与运行时安全评测](./paper-actfovea.md)
- [LIBERO 论文来源归档](../../sources/papers/rcl_awesome_wam_2306_03310_libero-benchmarking-knowledge-transfer-f.md)
- [LIBERO 项目仓库归档](../../sources/repos/libero-benchmark.md)

## 参考来源

- [论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf) 与 [arXiv 摘要](https://arxiv.org/abs/2306.03310)
- [官方 GitHub 仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [官方项目页](https://libero-project.github.io/) · [文档](https://lifelong-robot-learning.github.io/LIBERO/)
- [官方数据页](https://libero-project.github.io/datasets) · [Hugging Face 数据集](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)
- [RCL Awesome World-Action Models](https://github.com/rcl-robotics/Awesome-World-Action-Models)：第 031 项，仅作策展索引

