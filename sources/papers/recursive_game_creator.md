# Recursive Game Creator：面向体验优化的递归式游戏开发框架

> 来源归档；核查日期：2026-10-19。

- **类型：** paper
- **论文：** [Recursive Game Creator: An Agentic Product-Level Experience-Oriented Game Harness](https://arxiv.org/abs/2610.08621)
- **项目页：** <https://imbaldy.github.io/recursive-game-creator/>
- **官方代码：** <https://github.com/IMBALDY/RecursiveGameCreator>
- **Hugging Face：** <https://huggingface.co/papers/2610.08621>（论文索引页）
- **作者单位：** 香港大学（HKU）、深圳河套研究院（Shenzhen Loop Area Institute）
- **论文版本：** arXiv v1，2026-10-06
- **入库日期：** 2026-10-19
- **一句话说明：** 由 Designer、Builder、coding-native Player、Reviewer 四角色构成闭环，让代码代理通过可执行游玩、证据评估和反馈多轮改进游戏。
- **项目页归档：** [Recursive Game Creator 项目页](../sites/recursive-game-creator.md)
- **代码归档：** [RecursiveGameCreator 仓库](../repos/recursive-game-creator.md)
- **Wiki：** [Recursive Game Creator](../../wiki/entities/paper-recursive-game-creator.md)

## 核心摘录

1. **四角色闭环：** Designer 将用户需求与上一轮反馈转为体验假设、修改计划和验收目标；Builder 实现可玩的候选版本；Player 编写可复用策略并通过程序接口收集状态、动作、事件、进度、耗时和终止结果；Reviewer 综合轨迹、视觉信息及用户偏好，比较候选版本并给出下一轮修改反馈。
2. **体验证据：** Player 的状态接口让策略可重复执行，不必每一步都请求语言模型或从画面还原状态。策略轨迹用于估计成功率、难度、探索覆盖和卡点；论文提醒，代理游玩时间本身不能证明真人兴趣。
3. **GameCraft-Bench：** 在 45 个选定任务上运行三轮改进。GPT-6 Astra 配置下 Overall 从 R1 的 72.70 升至 R3 的 77.89（+5.19），比同模型基线 71.26 高 6.63。
4. **GameASG-Bench：** 47 个浏览器游戏任务中严格成功 25 个（53.2%）；平均 L1 源码检查通过率 98.3%，平均 L2 浏览器运行检查通过率 93.4%。
5. **开放情况：** 项目页链接公开官方仓库；仓库采用 MIT 许可证并含安装、运行说明。本次未发现 HF 论文页之外的独立数据集或模型权重入口。

## 对 Wiki 的映射

- 将论文、项目页与官方代码归纳到同一项目实体页。
- 可用于理解 coding agent 的执行式评测与多轮产品迭代；它评估游戏软件体验，不构成机器人或物理仿真验证。
