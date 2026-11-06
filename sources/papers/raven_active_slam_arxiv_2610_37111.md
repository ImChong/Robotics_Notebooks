# RAVEN: A Unified Active SLAM System for Rapid Autonomous Visual Exploration with Aerial Robots under Localization Uncertainty

- **类型：** 论文，arXiv 预印本 v1，2026-10-27
- **作者：** Xinyi Chen, Yichen Zhang, Chenxing Jiang, Shaojie Shen
- **机构：** 香港科技大学电子及计算机工程学系
- **原文：** https://arxiv.org/pdf/2610.37111v1
- **核对范围：** 摘要、系统概览、目标与效用评估、OFIF 主要公式、仿真与实机实验；全文其余内容尚待逐节复核。
- **通读：** 未完成
- **代码：** 论文表示将发布源码；尚未找到该项目的官方实现。

## 原文要点与 wiki 映射

1. 识别探索边界与主动回环候选，生成覆盖路线并插入至多一个回环候选；见 [RAVEN 实体](../../wiki/entities/paper-raven-active-slam.md)。
2. 候选分别用预计探索时间和考虑回环成功概率的位姿图信息评估，再按参数从 Pareto 前沿选路线；见实体的核心机制。
3. 在线 Fisher 信息场支持位姿信息快速查询，供位置与偏航规划使用；见实体的流程与机制。
4. 仿真和实机展示效率与精度折中；真机外部测量仅作评估，见实体的实验与局限。
5. §I、§V-C、§VI-C、§VII-A、§VIII-A2、§VIII-B2、表 II–III、图 9：用于实体页「与其他工作对比」（2026-11-05 补）。
