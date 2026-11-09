# Ai2 ToyBlocks: A controlled world for robot learning and evaluation

- **类型：** 官方博客 / 官方技术文档
- **中文标题：** ToyBlocks：可控的机器人学习与评测基础世界
- **作者机构：** Allen Institute for AI（Ai2）
- **发布时间：** 2026-11-06
- **原文：** https://huggingface.co/blog/allenai/toyblocks
- **官方项目：** https://allenai.github.io/ToyBlocks
- **官方代码：** https://github.com/allenai/ToyBlocks
- **官方文档：** https://allenai.github.io/ToyBlocks/docs/user_guide/index.html
- **检查日期：** 2026-11-09
- **核对范围：** 通读 Ai2 官方博客的系统架构、自动演示、两级模型、数据、仿真及真机评测、后续开放计划，核对 GitHub README 和 datagen/eval 入口；未独立运行仿真。
- **通读：** 已完成

## 原文要点
1. Isaac Sim/Isaac Lab 的可控桌面积木场景，用 PDDL 任务规格生成可规划的积木推、翻、堆叠操作。
2. 自动演示流水线包含 task generation → 环境与重置 → FF/PDDL 规划与技能执行 → 可选分离渲染 → LeRobot 合并与标注，不依赖人工遥操作逐条采集。
3. 作者报告约 85,336 条演示、近 1,600 小时、约 1.44 亿帧；这些是特定发布数据集口径，不是每个默认单 GPU quick start 会生成的规模。
4. 低层 ManiFlow 生成具身动作；高层可用 Molmo2 等视觉语言模型决策技能。仿真基准可隔离高低层 oracle 定位失败。
5. 论文/博文报告的仿真和物理机器人成功率分开统计；当前完整的 RL 训练集成仍列作未来工作，不能将演示数据开放等同 RL pipeline 全部可复现。

## 对 wiki 的映射
- [ToyBlocks](../../wiki/entities/toyblocks.md)
- [源码归档](../repos/allenai_toyblocks.md)
