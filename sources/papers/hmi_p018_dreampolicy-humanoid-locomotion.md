# DreamPolicy: A Unified World-model Policy for Scalable Humanoid Locomotion（DreamPolicy，HMI P018）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** DreamPolicy: A Unified World-model Policy for Scalable Humanoid Locomotion
- **短名：** DreamPolicy
- **类型：** paper / hmi-papers / Locomotion与运动先验
- **HMI ID：** P018
- **年份：** 2025
- **原文：** https://arxiv.org/abs/2505.18780
- **代码：** 无 / 见正文开源状态
- **项目页：** https://dreampolicy.github.io/
- **入库日期：** 2026-07-31
- **一句话说明：** 先采多地形专家数据训自回归扩散世界模型生成未来状态，再以目标条件 RL 学统一跟踪策略，减少混合地形重复奖励工程。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P018](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P018.md)

## 开源状态（步骤 2.5）

- **结论：** 截至策展日项目页未见训练代码入口（待再核）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

DreamPolicy想解决“每种地形一个专家”的扩展问题。作者先在五类地形上训练专用RL策略并收集本体状态、地形观测、动作和奖励，再训练一个地形条件的自回归扩散模型生成未来身体状态序列。统一策略不直接复制专家动作，而把生成的未来状态当作动态目标，通过goal-conditioned RL学会跟踪。

**对 wiki 的映射：** [`wiki/entities/paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md`](../../wiki/entities/paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md)

### 摘录 2

专家策略仍需要基础行走奖励和地形专用塑形，数据采集也发生在仿真中。因此本文减少的是**统一策略阶段**针对混合地形的重复奖励工程，不是完全消除前期专家、场景和奖励设计。数据覆盖哪些地形、专家在哪些状态成功，决定扩散模型能“梦到”什么。

**对 wiki 的映射：** [`wiki/entities/paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md`](../../wiki/entities/paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md)

### 摘录 3

模型在机器人自身坐标系中生成包含本体与地形信息的未来状态，避免依赖全局位置和航向。训练从teacher forcing逐步过渡到使用模型自身历史的自回归rollout，以减轻训练和推理的分布差。部署时扩散模型给出未来状态轨迹，统一策略读取当前观测和该轨迹，输出29维关节位置目标。

**对 wiki 的映射：** [`wiki/entities/paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md`](../../wiki/entities/paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md`](../../wiki/entities/paper-notebook-one-policy-but-many-worlds-a-scalable-unified-po.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
