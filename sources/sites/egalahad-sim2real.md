# sim2real 文档站与动作跟踪排行榜

> 来源归档

- **标题：** sim2real — A Lightweight and Modular Sim2sim and Sim2real Deployment Stack
- **类型：** project site / documentation / benchmark
- **项目文档：** <https://egalahad.github.io/sim2real/>
- **Leaderboard：** <https://egalahad.github.io/sim2real/leaderboard>
- **官方代码：** <https://github.com/EGalahad/sim2real>
- **论文 / 技术报告：** 项目 README 提供 2026 年 BibTeX 引用；本次核查未发现独立 arXiv 编号
- **开源状态：** 部署与评测代码公开；大型运行时检查点和第三方资产通过共享云盘分发，不随 Git 仓库保存
- **页面标注更新时间：** Leaderboard protocol 标注 2026-09-15
- **核查日期：** 2026-10-13
- **一句话说明：** EGalahad 的 sim2real 项目文档与 Unitree G1 多策略全身动作跟踪匹配评测入口；通过一致的参考动作、评测器和失败条件比较来自不同训练代码库的已发布策略。

## 页面核查

- 文档站覆盖安装、离线动作跟踪、PICO 遥操作、动作记录、外部策略运行、机器人 I/O 和 JetPack 5 部署说明。
- Leaderboard 默认使用 MotionDecode 的 Locomotion-80、Manipulation-48、Ground-60、Dance-40；保留 Legacy LAFAN-40、PHUMA-30、Root-90 作为可选数据集。
- 指标包含身体位置 / 姿态误差、全局根误差、GPU 训练时长、手腕位置 / 姿态误差、Tracking Return 和 Progress。
- 协议固定 seed 0、无运行时长上限，采用共同终止条件；策略仍保留各自 observation 与 action processing，因此它是 matched evaluation，不代表所有策略使用同一训练或部署实现。
- 项目仓库 README 列出 Mimic Lite v1.1、ROA、PPO、HEFT、HoloMotion、SONIC 三种发布项、GRIT v0.0.1、ScaleBFM XL/M、BFM-Zero、TeleopIt、Humanoid-GPT 和 TWIST2，共 15 个策略条目。
- 同一算法项目的版本共用一个详情节点；本库将 15 个策略条目映射到 10 个独立项目节点，避免把 checkpoint 变体误建成重复项目。

## 开源状态与资料边界

- **代码：** 已开源，见 [EGalahad/sim2real](https://github.com/EGalahad/sim2real)。
- **权重 / 资产：** README 指向共享云盘中的 checkpoints 与 third_party；不要将这些运行时产物误认为 Git 仓库内容。
- **评测数据：** 指标清单公开于仓库的 docs/src/data/leaderboard.json；协议与指标定义在 docs/leaderboard.mdx。
- **评测复现：** 官方提供 [Cross-Codebase Tracking Evaluation](https://egalahad.github.io/sim2real/tutorials/cross-codebase-evaluation/) 教程。

## 对 wiki 的映射

- [sim2real 项目与排行榜](../../wiki/entities/sim2real.md)
- [sim2real 代码归档](../repos/egalahad_sim2real.md)
- [Sim2Real 概念](../../wiki/concepts/sim2real.md)
