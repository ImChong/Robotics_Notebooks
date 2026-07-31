# Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild（Robust Perceptive Locomotion，HMI P012）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild
- **短名：** Robust Perceptive Locomotion
- **类型：** paper / hmi-papers / Locomotion与运动先验
- **HMI ID：** P012
- **年份：** 2022
- **原文：** https://arxiv.org/abs/2201.08117
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 用循环 Belief Encoder 融合带噪高程图与本体历史，使四足在外感知失效时仍能退回身体反馈、在野外稳健行走。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P012](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P012.md)

## 开源状态（步骤 2.5）

- **结论：** 论文未作为本库主复现入口挂代码；概念影响后续感知 loco 管线

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

常见感知运动管线把点云融合成2.5D高程图，再把地图高度送给控制器。问题是雪、植被、反光、遮挡、位姿漂移和视野外区域都会让地图出现假台阶或空洞。本文没有假设地图总是正确，而是让一个循环Belief Encoder把带噪地形观测与本体感觉历史融合成信念状态，控制器可以在外感知可靠时提前调整，在外感知异常时退回身体反馈。

**对 wiki 的映射：** [`wiki/entities/paper-robust-perceptive-locomotion-wild.md`](../../wiki/entities/paper-robust-perceptive-locomotion-wild.md)

### 摘录 2

第一阶段的教师在仿真中能看到无噪地形、接触和环境参数，用PPO得到高性能策略。第二阶段学生接收真机可得的本体感觉和被系统性破坏的高度采样：随机偏移模拟里程计漂移，大噪声和遮挡模拟传感器失效，局部错误模拟地图异常。循环编码器从历史构造belief，策略模仿教师动作；同时解码器要求belief重建无噪高度与特权状态。

**对 wiki 的映射：** [`wiki/entities/paper-robust-perceptive-locomotion-wild.md`](../../wiki/entities/paper-robust-perceptive-locomotion-wild.md)

### 摘录 3

行为克隆损失回答“下一步动作是否像教师”，重建损失回答“中间表示是否保留了与控制有关的环境信息”。编码器中的门控决定多少外感知进入belief，因此策略不必在每一帧都同等相信地图。

**对 wiki 的映射：** [`wiki/entities/paper-robust-perceptive-locomotion-wild.md`](../../wiki/entities/paper-robust-perceptive-locomotion-wild.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-robust-perceptive-locomotion-wild.md`](../../wiki/entities/paper-robust-perceptive-locomotion-wild.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
