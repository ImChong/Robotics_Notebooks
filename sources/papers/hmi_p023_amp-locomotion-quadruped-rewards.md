# Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions（AMP Locomotion，HMI P023）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions
- **短名：** AMP Locomotion
- **类型：** paper / hmi-papers / Locomotion与运动先验
- **HMI ID：** P023
- **年份：** 2022
- **原文：** https://arxiv.org/abs/2203.15103
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 在四足上证明：保留速度任务奖励、用短段犬类动作 AMP 即可替代大量手工步态塑形项，并部署到 Unitree A1。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P023](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P023.md)

## 开源状态（步骤 2.5）

- **结论：** 方法复用 AMP/ASE 生态；本篇以 A1 真机验证为主

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

这篇工作的重点不是再次介绍AMP，而是做一个很有工程意义的检验：四足步态奖励通常包含抬脚、落脚、躯干姿态、对称性和能耗等大量规则，能否只保留速度任务奖励，再让一小段动作数据负责运动风格？作者用约4.5秒德国牧羊犬动作训练Unitree A1，并把策略部署到真机。

**对 wiki 的映射：** [`wiki/entities/paper-amp-locomotion-quadruped-rewards.md`](../../wiki/entities/paper-amp-locomotion-quadruped-rewards.md)

### 摘录 2

作者先把约4.5秒的德国牧羊犬运动重定向到Unitree A1。重定向使用逆运动学求机器人关节角，再以前向运动学检查脚端等关键部位的位置，关节与末端速度由相邻帧差分得到。动作片段覆盖慢速踱步、快步、小跑和转向，它们不带“当前应跟踪多少速度”的逐帧标签，只向判别器提供自然四足运动的短时状态转移。

**对 wiki 的映射：** [`wiki/entities/paper-amp-locomotion-quadruped-rewards.md`](../../wiki/entities/paper-amp-locomotion-quadruped-rewards.md)

### 摘录 3

每个控制周期，Actor读取A1关节角、关节速度、机身方向、上一时刻动作，以及用户给出的前向速度、侧向速度和偏航角速度命令。速度命令范围覆盖后退到快速前进、横移和左右转向。Actor使用三层MLP输出十二个关节目标角，策略以30 Hz运行，底层PD控制器把目标角转换为电机力矩。真机执行后的关节和机身状态回到下一周期，形成速度命令到关节动作的闭环。

**对 wiki 的映射：** [`wiki/entities/paper-amp-locomotion-quadruped-rewards.md`](../../wiki/entities/paper-amp-locomotion-quadruped-rewards.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-amp-locomotion-quadruped-rewards.md`](../../wiki/entities/paper-amp-locomotion-quadruped-rewards.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
