# WorldArena 官方仓库

> 来源归档（WorldArena 维护团队官方代码与 Track 1 评测文档）

- **标题：** WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models
- **类型：** repo / benchmark
- **代码：** <https://github.com/tsinghua-fib-lab/WorldArena>
- **项目页：** <https://world-arena.ai/>
- **论文：** <https://arxiv.org/abs/2602.08971>
- **Track 1 指南：** <https://github.com/tsinghua-fib-lab/WorldArena/blob/main/assets/README_submission.md>
- **Track 1 指标实现：** <https://github.com/tsinghua-fib-lab/WorldArena/tree/main/video_quality>
- **Track 1 数据集：** <https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0>
- **开源状态：** 已公开评估代码、Track 1/2 指南及测试资源；官方指南称 Challenge 1.0 已结束，GT 与评估资源已公开，支持本地运行。
- **入库日期：** 2026-10-16（原仓库入口归档：2026-07-30）
- **一句话说明：** WorldArena 1.0 将视频感知质量与具身功能效用纳入同一评测框架；Track 1 的 16 项指标分为六个维度，Track 2 检查模型作为数据引擎和策略评估器的效用。

## Track 1：Video Quality

- **数据：** RoboTwin 2.0 Clean-50 子集；官方评测集为 Hugging Face 数据仓库中的 `test_dataset`。
- **输入：** 首帧与文本指令，或首帧与动作序列；评估时对照真值视频。
- **输出要求：** 推荐分辨率不低于 640×480；文本驱动生成固定 121 帧；动作驱动长度与给定动作轨迹对齐；24 fps。
- **指标维度：** visual quality、motion quality、content consistency、physics adherence、3D accuracy、controllability，共 16 项基础指标。
- **总分：** 16 项归一化基础指标缩放到 0–100 后取算术平均；正式分析仍应读分项。

## 仓库阅读入口

- `video_quality/README.md`：Track 1 指标与本地评估管线。
- `assets/README_submission.md`：两条赛道的评测指南、输入和生成要求。
- `embodied_task/`：Track 2 功能评估代码。
- 仓库 README 将 Track 2 分为 Data Engine、Policy Evaluator，并在主框架中纳入 Action Planner。

## 与相邻基准的关系

| 基准 | 差异 |
|------|------|
| [EWMBench](../../wiki/entities/ewmbench.md) | 聚焦机器人操作视频的场景、末端运动与语义一致性 |
| [WorldOlympiad](../../wiki/entities/worldolympiad.md) | 聚焦长视频物理规则、3D几何与跨片段交互诊断 |

三者的测试域、指标和协议不同，不宜仅按汇总分直接排名。

## 对 wiki 的映射

- [WorldArena（具身世界模型评测）](../../wiki/entities/worldarena.md)
