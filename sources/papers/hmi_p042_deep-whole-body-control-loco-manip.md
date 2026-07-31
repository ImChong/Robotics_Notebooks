# Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion（Deep Whole-Body Control，HMI P042）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion
- **短名：** Deep Whole-Body Control
- **类型：** paper / hmi-papers / LocoManip
- **HMI ID：** P042
- **年份：** 2022
- **原文：** https://arxiv.org/abs/2210.10044
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 用 Advantage Mixing 平衡移动与操作梯度，并配合在线适应估计环境变化，使统一策略在共享身体上协调两类行为。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P042](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P042.md)

## 开源状态（步骤 2.5）

- **结论：** 部分/社区复现线索需按原文与项目页再核

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

四足加机械臂常用两个独立控制器：腿追速度，臂追末端，两者通过基座扰动被动耦合。本文用一个策略同时读取腿、臂、基座、末端目标和移动命令，直接输出18个关节目标。真正的难点不是网络维度，而是训练早期操作回报和移动回报会把策略拉向不同局部最优。

**对 wiki 的映射：** [`wiki/entities/paper-deep-whole-body-control-loco-manip.md`](../../wiki/entities/paper-deep-whole-body-control-loco-manip.md)

### 摘录 2

策略输入包含基座和关节状态、足端接触、上一动作、末端位置姿态命令、机身速度命令以及环境latent，输出腿和臂的目标关节位置。作者分别计算manipulation advantage与locomotion advantage：训练初期让臂动作更多受操作优势更新、腿动作更多受移动优势更新，使两个子任务先形成有效探索；随后逐渐混合总优势，让全身学会协调。没有这一过程时，策略很容易停在原地追末端，因为走路初期会暂时降低操作回报。

**对 wiki 的映射：** [`wiki/entities/paper-deep-whole-body-control-loco-manip.md`](../../wiki/entities/paper-deep-whole-body-control-loco-manip.md)

### 摘录 3

统一策略的收益不是形式上的“一个网络”，而是腿可以移动和倾斜基座扩大机械臂工作空间，手臂受力时腿也能主动补偿。论文通过独立策略、未协调单策略和完整方法对比，说明协调来自共享状态与联合优化，而不是简单拼接输出。

**对 wiki 的映射：** [`wiki/entities/paper-deep-whole-body-control-loco-manip.md`](../../wiki/entities/paper-deep-whole-body-control-loco-manip.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-deep-whole-body-control-loco-manip.md`](../../wiki/entities/paper-deep-whole-body-control-loco-manip.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
