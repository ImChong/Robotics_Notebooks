# GR00T-Dreams: Synthetic Trajectory Generation for Humanoid Robot Learning（GR00T-Dreams，HMI P068）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** GR00T-Dreams: Synthetic Trajectory Generation for Humanoid Robot Learning
- **短名：** GR00T-Dreams
- **类型：** paper / hmi-papers / 世界模型、VLA与Agent
- **HMI ID：** P068
- **年份：** 2025
- **原文：** https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/
- **代码：** 无 / 见正文开源状态
- **项目页：** https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/
- **入库日期：** 2026-07-31
- **一句话说明：** NVIDIA 合成轨迹 blueprint：少真实遥操 post-train Cosmos → 语言生成视频 dreams → 筛选 → IDM 标动作 → 与真数据共训 VLA。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P068](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P068.md)

## 开源状态（步骤 2.5）

- **结论：** blueprint/参考工作流；组件开源边界以 NVIDIA 博客与 Cosmos/GR00T 各仓为准

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

GR00T-Dreams是NVIDIA在2025年发布的一套blueprint/参考工作流，不是一篇单一模型论文。它的出发点是真机遥操太贵：先用少量某本体、某环境的真实示范教Cosmos这台机器人和任务的外观/运动，再用文字生成新场景和新动作视频，最后把合格视频反推成可训练的机器人动作。

**对 wiki 的映射：** [`wiki/entities/paper-gr00t-dreams-synthetic-trajectories.md`](../../wiki/entities/paper-gr00t-dreams-synthetic-trajectories.md)

### 摘录 2

1. 先采集少量真实遥操轨迹，用来post-train Cosmos Predict-2，给世界模型注入目标机器人的外观、运动约束和环境。 2. 从一张初始图像和新语言指令生成大量2D视频“dreams”，扩展物体、背景和行为组合。 3. 用Cosmos Reason判断动作是否成功、场景是否合理，过滤明显失败或幻觉视频。 4. IDM读“前帧 + 后帧”，预测中间的3D动作段，把纯像素视频转成带动作标签的neural trajectory。 5. 将神经轨迹与真实数据共同训练或后训练VLA，再回到真机检验。

**对 wiki 的映射：** [`wiki/entities/paper-gr00t-dreams-synthetic-trajectories.md`](../../wiki/entities/paper-gr00t-dreams-synthetic-trajectories.md)

### 摘录 3

每条合成样本至少应保存初始图像、语言条件、生成视频、筛选分数、IDM动作块、目标本体schema和来源模型版本。只有视频而没有动作对齐，不能训练VLA；只有IDM动作而没有可追溯视频和筛选记录，也无法定位标签错误。合成数据与真实数据的batch比例、动作归一化和任务去重会直接影响后训练结果。

**对 wiki 的映射：** [`wiki/entities/paper-gr00t-dreams-synthetic-trajectories.md`](../../wiki/entities/paper-gr00t-dreams-synthetic-trajectories.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-gr00t-dreams-synthetic-trajectories.md`](../../wiki/entities/paper-gr00t-dreams-synthetic-trajectories.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
