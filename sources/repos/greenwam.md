# greenvla/GreenWAM

> 来源归档

- **标题：** GreenWAM: A World Foundation Model for Robotics
- **类型：** 官方代码仓库
- **链接：** https://github.com/greenvla/GreenWAM
- **项目页：** https://greenvla.github.io/GreenWAM/
- **论文：** https://arxiv.org/abs/2610.18774
- **权重：** https://huggingface.co/SberRoboticsCenter/GreenWAM
- **核查日期：** 2026-10-15
- **一句话说明：** 发布 GreenWAM 的四个基准 checkpoint 与对应的策略服务端、benchmark clients 和 WorldArena 评估代码。

## README 要点
- **共享动作空间：** 20 类机器人本体被映射到 79 个共享解剖关节槽位。
- **基座：** README 称模型基于 Cosmos 3 Edge，并联合训练视觉、动作条件与 world-action 模式。
- **评测 checkpoint：** RoboTwin 2.0、LIBERO / LIBERO-plus、CALVIN ABC→D、WorldArena 1.0。
- **入口示例：** RoboTwin 策略通过 `greenwam.infer.robotwin_policy_server` 启动；CALVIN 与 LIBERO 使用各自 server；WorldArena 通过 `benchmarks/worldarena/rollout.sh` 运行。
- **依赖：** Git LFS、FFmpeg、CUDA/uv 锁定环境；checkpoint 另依赖 Wan2.2 VAE、Cosmos 3 Edge tokenizer 和各基准独立环境。
- **README 报告结果：** RoboTwin 2.0 mean 67.54（clean 86.98、randomized 48.10）；LIBERO 97.4、zero-shot LIBERO-plus 86.8；CALVIN 平均链长 4.64；WorldArena EWMScore 63.31。数值是上游报告，复现实验应沿用各项目的评测协议。

## 开源边界
代码、评估入口和四个 benchmark checkpoint 已链接公开。不要把仓库的评测实现等同于所有上游仿真器、数据或资产；复现前按 README 安装每个基准的环境与模型组件。

## 对 wiki 的映射
- 实体页：[GreenWAM](../../wiki/entities/paper-greenwam.md)
- 项目页：[GreenWAM 项目归档](../sites/greenwam-project.md)
- 论文：[arXiv:2610.18774](../papers/greenwam_arxiv_2610_18774.md)
- WAM 概念：[世界–动作模型](../../wiki/concepts/world-action-models.md)
