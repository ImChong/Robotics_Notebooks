# RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots · 方法页一手资料复核

- **原文：** <https://arxiv.org/html/2406.02523v1>
- **通读：** 已完成
- **核对范围：** 通读 v1 §I–VI 与附录 VII–IX，核对场景布局与风格、可交互资产、任务定义及演示扩展，并核对官网对 v1 和后续版本的区分；未运行仿真。
- **入库日期：** 2026-11-26
- **项目主页及源码入口：** <https://robocasa.ai/>；主页提供官方 GitHub 与文档入口，本次记录初版机制，不声称当前仓库默认配置复刻 v1。

初版用布局与样式组合得到 120 个厨房场景并提供可交互资产和 100 个任务。生成模型辅助资产与候选任务，但仍需转换为 MuJoCo 可用的模型，人工筛选复合任务并实现环境和判据，另用 MimicGen 扩展示范。官网现同时介绍后续 RoboCasa365，不能混用两版规模。

- [关联方法页](../../wiki/methods/robot-simulation-scene-generation.md)
