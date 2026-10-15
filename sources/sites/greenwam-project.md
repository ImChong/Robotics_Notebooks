# GreenWAM 项目页

> 来源归档（官方项目页、论文与权重入口）

- **标题：** GreenWAM: A World Foundation Model for Robotics
- **项目页：** https://greenvla.github.io/GreenWAM/
- **论文：** https://arxiv.org/abs/2610.18774
- **代码：** https://github.com/greenvla/GreenWAM
- **权重：** https://huggingface.co/SberRoboticsCenter/GreenWAM
- **组织：** Brain Team、Sber Robotics Center
- **核查日期：** 2026-10-15
- **一句话说明：** 一个跨 20 类机器人本体、同时覆盖世界预测和策略生成的机器人 world foundation model。

## 项目页与开放状态
- 项目页链接至技术报告、GitHub 代码和 Hugging Face 模型。
- 官方仓库 README 说明，公开内容包括评测代码、策略服务端、benchmark clients、WorldArena rollout / evaluator / scoring，以及 RoboTwin、LIBERO、CALVIN、WorldArena 四个基准的 checkpoint。
- checkpoint 依赖 Wan2.2 VAE 与固定 revision 的 Cosmos 3 Edge text tokenizer；仿真器和 evaluator 需要各自安装环境。仓库开放不代表所有 benchmark 数据、资产和运行环境都打包在仓库内。
- 论文中的真机人形结果属于报告结果；公开 README 列出的四个 checkpoint 面向上述 benchmark，不能据此推断真机部署包也已发布。

## 关联归档
- [论文来源](../papers/greenwam_arxiv_2610_18774.md)
- [官方代码与运行入口](../repos/greenwam.md)
- [wiki 实体](../../wiki/entities/paper-greenwam.md)
