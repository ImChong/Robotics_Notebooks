# Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning（Denoising World Model Locomotion，HMI P017）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning
- **短名：** Denoising World Model Locomotion
- **类型：** paper / hmi-papers / Locomotion与运动先验
- **HMI ID：** P017
- **年份：** 2025
- **原文：** https://arxiv.org/abs/2408.14472
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 用去噪世界模型在复杂地形上学习人形运动表征/策略，强调对噪声观测与地形不确定性的鲁棒性。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P017](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P017.md)

## 开源状态（步骤 2.5）

- **结论：** 以 HMI/项目页再核为准

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

标题里的World Model很容易让人联想到根据动作预测未来视频或潜在轨迹，但本文的方法并不是这类规划模型。DWL的核心是一个循环编码器-解码器：把加入遮挡、mask和domain randomization噪声的观测历史编码为latent，再重建训练时可见的真实状态；策略直接从这个latent输出关节目标，并与PPO联合训练。

**对 wiki 的映射：** [`wiki/entities/paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md`](../../wiki/entities/paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md)

### 摘录 2

仿真先从特权状态构造受污染观测，循环编码器根据历史得到`z_t`，解码器重建无噪状态。去噪损失由状态重建误差和latent稀疏正则组成；actor和critic同时用策略损失、价值损失学习。总目标把“估计什么信息”和“哪些信息能提高回报”耦合起来，而不是先训练一个独立估计器再冻结。

**对 wiki 的映射：** [`wiki/entities/paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md`](../../wiki/entities/paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md)

### 摘录 3

部署actor只使用本体传感历史，不读取训练时的高度扫描；动作是12维关节位置目标，经PD转成力矩。奖励仍包含速度跟踪、周期步态、足端轨迹和正则项，所以性能提升不能完全归因于去噪表示。论文还把当前奖励加入观测/表示设计，这一点复现时要核对真机是否能用同一定义实时计算，避免无意引入仿真特权量。

**对 wiki 的映射：** [`wiki/entities/paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md`](../../wiki/entities/paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md`](../../wiki/entities/paper-notebook-advancing-humanoid-locomotion-mastering-challeng.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
