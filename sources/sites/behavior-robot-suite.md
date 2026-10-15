# BEHAVIOR Robot Suite 项目页

> 来源归档（Stanford 官方项目页与合作方介绍）

- **标题：** BEHAVIOR Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities
- **项目页：** https://behavior-robot-suite.github.io/
- **合作方介绍：** https://galaxea-dynamics.com/blogs/robotics-applications/stanford-galaxea-brs-household-robotics
- **论文：** https://arxiv.org/abs/2503.05652
- **算法代码：** https://github.com/behavior-robot-suite/brs-algo
- **机器人控制代码：** https://github.com/behavior-robot-suite/brs-ctrl
- **数据：** https://huggingface.co/datasets/behavior-robot-suite/data
- **会议：** CoRL 2025
- **核查日期：** 2026-10-15
- **一句话说明：** Stanford 的家庭全身移动操作套件，以 Galaxea R1、JoyLo 低成本遥操作和 WB-VIMA 模仿学习策略为核心。

## 项目页核查
- Stanford 项目页列出论文、算法代码、机器人代码和数据集入口；GitHub 算法与硬件仓库均可访问，Hugging Face 提供数据。
- **JoyLo：** 以 Joy-Con 与双臂连杆组成全身遥操作接口；项目页报告硬件成本低于 500 美元。
- **WB-VIMA：** 使用本体关节间层级依赖，自回归预测全身动作，并以 self-attention 聚合多模态观测。
- **机器人：** Galaxea R1 轮式移动底盘、双臂和柔性 4-DoF 躯干。
- **评测：** 项目页报告五类真实家庭长时程任务。其显示的倍数结果来自论文和项目方评测，比较时须保留原始指标和基线协议。

## 关联归档
- [论文来源](../papers/behavior_robot_suite_arxiv_2503_05652.md)
- [算法、控制代码与数据入口](../repos/behavior-robot-suite.md)
- [wiki 实体](../../wiki/entities/behavior-robot-suite.md)
