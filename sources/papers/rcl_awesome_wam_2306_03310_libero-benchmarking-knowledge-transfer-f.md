# LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning

> 来源归档（论文原文 + 官方项目资料 + RCL Awesome World-Action Models 清单）

- **标题：** LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning
- **作者：** Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone
- **类型：** paper / robot-manipulation benchmark
- **出处：** NeurIPS 2023 Datasets and Benchmarks Track
- **arXiv：** <https://arxiv.org/abs/2306.03310>
- **NeurIPS 页面：** <https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html>
- **论文 PDF：** <https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf>
- **代码：** <https://github.com/Lifelong-Robot-Learning/LIBERO>
- **项目页：** <https://libero-project.github.io/>
- **文档：** <https://lifelong-robot-learning.github.io/LIBERO/>
- **官方数据页：** <https://libero-project.github.io/datasets>
- **Hugging Face 数据集：** <https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets>
- **策展列表：** [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models)，条目 031/564，分组 *Benchmarks & simulators*
- **一句话说明：** 提出用于终身机器人操作学习的 LIBERO 基准，包含 130 个语言条件任务与人类遥操作演示数据，用于分析知识迁移、策略结构和持续学习方法。
- **沉淀到 wiki：** [论文详情页](../../wiki/entities/libero-benchmark.md)

## 论文要点

LIBERO 将机器人终身学习中的知识迁移问题拆成可控的任务分布变化：空间关系、物体类别、任务目标，以及这些因素的混合变化。论文提供四个任务套件（共 130 项任务）和人类遥操作演示，并比较视觉-运动策略架构及终身学习算法。

论文报告的主要观察包括：在其前向迁移实验中，顺序微调优于所比较的终身学习算法；没有一种视觉编码架构在所有知识迁移类型上都占优；直接进行监督式预训练可能损害后续终身学习表现。具体任务定义、设置与结果以论文为准。

## 资料入口说明

- **论文页 / PDF / arXiv：** 论文正文、实验协议和结论的一手来源。
- **GitHub：** 官方基准实现、训练和评估脚本；README 提供演示数据下载脚本。
- **项目页 / 文档：** 基准介绍、任务说明及数据入口。
- **数据页 / Hugging Face：** 官方列出的演示数据下载入口；GitHub README 也说明可通过下载脚本选择 Hugging Face 来源。
- **RCL 清单：** 仅为策展索引，贡献摘要不替代论文正文。

## 对 wiki 的映射

- 论文详情页：[paper-rcl-2306-03310-libero-benchmarking-knowledge-transfer-for-lifel](../../wiki/entities/libero-benchmark.md)
- 基准工程页：[libero-benchmark](../../wiki/entities/libero-benchmark.md)
- 项目仓库归档：[libero-benchmark source](../repos/libero-benchmark.md)
- RCL 清单索引：[RCL catalog](rcl_awesome_wam_catalog.md)
