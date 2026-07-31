# Dream to Control: Learning Behaviors by Latent Imagination（Dreamer，HMI P064）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Dream to Control: Learning Behaviors by Latent Imagination
- **短名：** Dreamer
- **类型：** paper / hmi-papers / 世界模型、VLA与Agent
- **HMI ID：** P064
- **年份：** 2019
- **原文：** https://arxiv.org/abs/1912.01603
- **代码：** https://github.com/danijar/dreamer
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 从历史学习潜在转移，再在短时想象轨迹中训练 Actor-Critic，从而用更少真实交互学习像素控制行为。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P064](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P064.md)

## 开源状态（步骤 2.5）

- **结论：** 已开源（danijar/dreamer）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

PlaNet有了潜动力学，但在环境的每一步仍要用CEM评估上千组动作。Dreamer的改变是把这项在线搜索成本提前到训练阶段：它从真实经验学RSSM，再从经验对应的潜状态出发，在模型里想象大量未来，训一个actor和critic。真正与环境交互时，只需前向跑actor，不做规划搜索。

**对 wiki 的映射：** [`wiki/entities/paper-dreamer-latent-imagination.md`](../../wiki/entities/paper-dreamer-latent-imagination.md)

### 摘录 2

世界模型包含图像编码/重建、RSSM潜转移和奖励预测。它用replay buffer中的真实序列训练，不依赖actor对世界的猜测当监督信号。学行为时固定世界模型，从编码后的真实潜状态出发，actor采样动作，RSSM预测后续状态和奖励，critic估计每个状态的长期价值。因此世界模型学“动作会把世界带到哪里”，actor学“为了高回报应该选哪个动作”，两者不是一个损失。

**对 wiki 的映射：** [`wiki/entities/paper-dreamer-latent-imagination.md`](../../wiki/entities/paper-dreamer-latent-imagination.md)

### 摘录 3

真实环境每产生新图像、动作和奖励，就写入replay并继续更新世界模型；执行时历史经后验编码成当前belief，actor直接输出下一动作。与PlaNet不同，部署没有CEM内循环，重新适应环境主要依靠belief被新观测修正，以及后续训练用新数据更新模型和actor。单个episode内actor不会临时搜索一条全新计划。

**对 wiki 的映射：** [`wiki/entities/paper-dreamer-latent-imagination.md`](../../wiki/entities/paper-dreamer-latent-imagination.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-dreamer-latent-imagination.md`](../../wiki/entities/paper-dreamer-latent-imagination.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
