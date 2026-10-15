# BEHAVIOR Robot Suite 官方实现与数据

> 来源归档：同一 BRS 项目的算法仓库、机器人控制仓库与数据集入口

- **项目页：** https://behavior-robot-suite.github.io/
- **论文：** https://arxiv.org/abs/2503.05652
- **算法代码：** https://github.com/behavior-robot-suite/brs-algo
- **机器人控制代码：** https://github.com/behavior-robot-suite/brs-ctrl
- **JoyLo 设计文件：** 由官方项目页链接至公开硬件资源
- **训练数据：** https://huggingface.co/datasets/behavior-robot-suite/data
- **核查日期：** 2026-10-15
- **一句话说明：** BRS 的 WB-VIMA 算法、JoyLo / R1 控制代码与训练数据分别公开在官方仓库和 Hugging Face。

## 代码与数据边界
- `brs-algo` 是论文官方算法代码，GitHub 显示 MIT License；仓库包含训练与策略相关实现。
- `brs-ctrl` 是官方硬件控制代码，用于连接 JoyLo 和 Galaxea 机器人，含实时控制与数据采集 / 部署接口。
- 项目页还链接公开训练数据与 JoyLo 设计资源。代码可访问性不等于任何机器人均可直接运行；论文实验基于 R1，迁移本体需适配底盘、躯干、机械臂和夹爪接口。
- 论文深读已在 Robot Learning Paper Notebooks 单独维护；此处只保留官方仓库身份与本仓库所需复现入口。

## 对 wiki 的映射
- 实体页：[BEHAVIOR Robot Suite](../../wiki/entities/behavior-robot-suite.md)
- 项目页：[BRS 项目页](../sites/behavior-robot-suite.md)
- 论文：[arXiv:2503.05652](../papers/behavior_robot_suite_arxiv_2503_05652.md)
- 任务：[Loco-Manipulation / Mobile Manipulation 辨析](../../wiki/tasks/loco-manipulation.md)
