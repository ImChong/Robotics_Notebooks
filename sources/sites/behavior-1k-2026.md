# Robot Learning Collective｜2026 BEHAVIOR Challenge 方案官网

- **项目主页：** https://robot-learning-collective.github.io/behavior-1k-2026/
- **技术报告：** https://arxiv.org/abs/2611.14432
- **代码：** https://github.com/Robot-Learning-Collective/behavior-1k-solution-2026
- **权重：** https://huggingface.co/IliaLarchenko/BEHAVIOR2026VLA
- **训练数据：** https://huggingface.co/datasets/behavior-1k/2026-challenge-demos
- **核对日期：** 2026-11-12
- **核对范围：** 核对官方项目发布页、GitHub README 的论文、代码、模型与数据链接，并与技术报告对照；学习策略的分数与混合 Agent 方案分开记录。
- **通读：** 已完成

本方案**既有单策略**（100 项任务均可使用），也有面向 14 项弱项的额外 agent 分流。纯 VLA 本地 Q-score 0.392；最终公开排行榜混合方案 Q-score 0.446。作者公开了 VLA 权重、训练及推理脚本，但未公开 agent 的冻结练习会话，完整混合方案不能仅凭仓库复现。

- [论文归档](../papers/behavior_2026_solution_arxiv_2611_14432.md)
- [源码归档](../repos/behavior-1k-solution-2026.md)
- [wiki 主实体](../../wiki/entities/paper-behavior-2026-one-policy-100-tasks.md)
