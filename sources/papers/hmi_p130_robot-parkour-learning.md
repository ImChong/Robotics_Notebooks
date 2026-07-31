# Robot Parkour Learning（Robot Parkour Learning，HMI P130）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Robot Parkour Learning
- **短名：** Robot Parkour Learning
- **类型：** paper / hmi-papers / Locomotion与运动先验
- **HMI ID：** P130
- **年份：** 2023
- **原文：** https://arxiv.org/abs/2309.05665
- **代码：** https://github.com/ZiwenZhuang/parkour
- **项目页：** https://robot-parkour.github.io/
- **入库日期：** 2026-07-31
- **一句话说明：** 用直接配点启发的软→硬动力学约束课程先让策略发现可行动作，再蒸馏成接收深度的单一视觉四足跑酷策略。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P130](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P130.md)

## 开源状态（步骤 2.5）

- **结论：** 已开源（与 Extreme Parkour 同仓生态，但论文问题设定不同）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

Robot Parkour解决的是一个探索难题：攀高台、跨大沟、钻低洞和侧身穿缝都需要短时间内做出极端动作，但如果障碍从训练开始就是不可穿透的，随机策略几乎拿不到前进奖励；为每项技能手写一套动作模板又很难扩展。作者用“先让障碍可穿透，再恢复真实碰撞”的课程替代参考动作。

**对 wiki 的映射：** [`wiki/entities/paper-robot-parkour-learning.md`](../../wiki/entities/paper-robot-parkour-learning.md)

### 摘录 2

系统分别训练攀爬、跨沟、低姿穿越、侧身挤过和奔跑专家。第一阶段障碍允许穿透，碰撞点进入障碍的深度形成连续惩罚，策略即使还不会完整越障，也能从向前运动与较小穿透中获得梯度。课程逐渐提高障碍和穿透约束难度。第二阶段再换成不可穿透的硬碰撞环境细调，让动作满足真实接触动力学。整个过程只使用简单的前进、能耗和存活类奖励，没有动物参考动作，也没有AMP判别器。

**对 wiki 的映射：** [`wiki/entities/paper-robot-parkour-learning.md`](../../wiki/entities/paper-robot-parkour-learning.md)

### 摘录 3

专家训练时可以看到特权地形和物理状态，最后通过DAgger蒸馏成一个循环视觉策略。部署策略输入机载深度与本体历史，自动判断应当爬、跳、钻还是侧身，不需要外部技能切换器。作者还专门模拟深度缺失、噪声和延迟，并让视觉预处理尽量匹配真机相机。

**对 wiki 的映射：** [`wiki/entities/paper-robot-parkour-learning.md`](../../wiki/entities/paper-robot-parkour-learning.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-robot-parkour-learning.md`](../../wiki/entities/paper-robot-parkour-learning.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
