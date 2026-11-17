# MolmoB0T 官方项目页核查（Ai2）

- **网址：** https://allenai.github.io/MolmoBot/
- **官方博客：** https://allenai.org/blog/molmobot-robot-manipulation
- **论文：** https://arxiv.org/abs/2603.16861
- **代码（Models）：** https://github.com/allenai/MolmoBot
- **代码（Data）：** https://github.com/allenai/molmospaces
- **数据集：** https://huggingface.co/datasets/allenai/molmobot-data
- **模型集合：** https://huggingface.co/collections/allenai/molmobot-models
- **核查日期：** 2026-11-17
- **核对范围：** 官方项目页的摘要、论文/Blog/Code/Models/Dataset 入口、数据生成段、DROID 真机实验分母与任务分布、移动平台演示及开源资源；对照论文 v3 §§3–6。
- **通读：** 已完成
- **开源结论：** 已开源。官方提供机器人模型训练/推理与真机推理脚本、MolmoSpaces 数据生成工具、模型 checkpoint、数据集。可核验代码入口不等于已在本知识库环境执行训练或真机复现。

## 页面事实与解释

- 官方报告仿真数据约 1.7M demonstrations、5,704h；页面强调跨不同视角、场景、灯光的零样本迁移。
- DROID 真机：40 项 pick-and-place（四环境 × 十项），每模型 120 回合；MolmoBot (H=1) 79.2%，π₀.₅ 39.2%；属于作者受控协议的结果。
- 作者宣传的“entirely in simulation / no real-world data”针对机器人操作策略训练；**MolmoBot 使用已经基于真实图像与视频预训练的 Molmo2-4B**，见论文 §4.1 与 Molmo2 原论文。
- 页面有模型/数据代码及视频演示，但暂未在本地跑通模型，不把“代码存在”写成“已复现”。

## 对知识库的映射

- [主论文原文归档](../papers/molmobot_arxiv_2603_16861.md)
- [官方代码核查](../repos/allenai-molmobot.md)
- [数据集原始资料](../datasets/allenai-molmobot-data.md)
- [独立论文详情](../../wiki/entities/paper-molmobot.md)
