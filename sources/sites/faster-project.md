# FASTER 项目页

> 来源归档（site）

- **标题：** FASTER: Value-Guided Sampling for Fast RL
- **项目页：** <https://pd-perry.github.io/faster/>
- **论文：** <https://arxiv.org/abs/2604.19730>
- **代码：** Robomimic <https://github.com/alexanderswerdlow/faster>；π0.5 / VLA <https://github.com/alexanderswerdlow/faster_vla>
- **机构：** Stanford University
- **入库日期：** 2026-10-07
- **一句话说明：** 在扩散策略去噪过程中用价值引导过滤动作候选，保留测试时多采样收益并降低计算成本。

## 项目页核查

- **代码状态：** 项目页分别提供基础 RL / Robomimic 代码和 VLA 代码两个入口；均已公开。
- **方法解释：** 以 MDP 描述多个噪声动作候选的逐步去噪及筛选；critic 可在完整去噪前估计候选的下游价值。
- **论文主张：** 长时程操纵 online 与 batch-online RL 上提高底层策略；预训练 VLA 实验在性能相当时降低训练和推理计算。

## 复现入口

VLA 代码仓库单独组织训练进程与 LIBERO 环境进程，README 给出的运行入口包含 FASTER-EXPO 与 plain EXPO；Robomimic 仓库则提供 online / batch-online 多种实验脚本。
