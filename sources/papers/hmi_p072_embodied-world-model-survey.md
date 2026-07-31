# A Comprehensive Survey on World Models for Embodied AI（Embodied World Model Survey，HMI P072）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** A Comprehensive Survey on World Models for Embodied AI
- **短名：** Embodied World Model Survey
- **类型：** paper / hmi-papers / 世界模型、VLA与Agent
- **HMI ID：** P072
- **年份：** 2025
- **原文：** https://arxiv.org/abs/2510.16732
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 综合整理具身世界模型的表示、训练目标、规划/策略用法与评测，帮助区分「会预测画面」与「能支撑决策」。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P072](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P072.md)

## 开源状态（步骤 2.5）

- **结论：** 综述

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

“世界模型”已同时指向Dreamer式任务潜动力学、自动驾驶占据预测、视频生成模型和通用互动模拟器。这篇综述用三条互相独立的轴重新组织它们：是否与具体决策任务耦合，未来是逐步递推还是全局差分预测，世界状态又用什么空间表示。这比按“扩散/Transformer”列模型更能反映它们能否接入机器人闭环。

**对 wiki 的映射：** [`wiki/entities/paper-embodied-world-model-survey.md`](../../wiki/entities/paper-embodied-world-model-survey.md)

### 摘录 2

Decision-coupled模型紧贴某个环境、奖励或策略，目标是为规划、价值估计或策略学习产生足够准确的可操作预测，PlaNet和Dreamer是典型例子。General-purpose模型强调跨场景、跨任务的高保真未来生成，可以是数据引擎、互动模拟器或下游agent的环境。前者可以画面不漂亮但控制有用，后者可以视频逼真但尚不保证动作-物理因果准确。评估时不能拿FVD代替任务成功率，也不能只用单任务回报宣称通用世界模型。

**对 wiki 的映射：** [`wiki/entities/paper-embodied-world-model-survey.md`](../../wiki/entities/paper-embodied-world-model-survey.md)

### 摘录 3

Sequential simulation/inference从当前状态生成下一状态，再将预测喂回继续展开；它天然适合MPC和任意时域，但每步偏差会累积。Global difference prediction直接从初始状态与时间/动作条件预测远期差分或多个未来，可并行、减少自回归漂移，但对任意长度及中间过程的表达弱。这两类选择与控制时域、计算预算和是否需要中间接触细节直接相关。

**对 wiki 的映射：** [`wiki/entities/paper-embodied-world-model-survey.md`](../../wiki/entities/paper-embodied-world-model-survey.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-embodied-world-model-survey.md`](../../wiki/entities/paper-embodied-world-model-survey.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
