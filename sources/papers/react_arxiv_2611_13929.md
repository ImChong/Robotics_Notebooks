# ReACT｜非对称负载下的人形机器人速度跟踪

- **英文标题：** ReACT: Reflection-Consistent Modulation and Cross-Load Consistency for Humanoid Velocity Tracking Under Asymmetric Loads
- **论文：** https://arxiv.org/abs/2611.13929
- **全文：** https://arxiv.org/html/2611.13929v1
- **版本：** arXiv v1，2026-11-10
- **作者：** Xianghui Wang、Aoran Fu、Liyou Luo、Shanze Wang、Xiaoyu Shen、Dan Zhang、Wei Zhang
- **研究对象：** 单手持物造成非对称外负载的 Unitree G1 速度指令跟踪
- **入库日期：** 2026-11-12
- **核对范围：** arXiv 官方 v1 全文的摘要、问题设置、reflection-consistent 网络结构、跨负载一致性训练机制、仿真基准、G1 真机导航结果与局限；结果与比较口径逐项对照表格。未运行作者系统。
- **通读：** 已完成

## 研究问题与核心机制

普通人形速度策略的左右对称归纳偏置有利于迁移，却可能无法充分利用单手负载导致的外力/力矩不对称。ReACT 用**精确遵循左右镜像变换的特征调制**和**跨负载轨迹一致性**，而不是单纯增加负载随机化，改善同一速度命令在空载、左手和右手载荷条件下的行为一致性。

- 策略观测约 **106 维**，包括关节/姿态、速度命令与通过 15 帧本体历史估计出的 6 维外部 wrench；动作覆盖双腿和腰部的 **15 个关节**，双臂的持物姿态在外层指定。
- **Reflection-consistent modulation**：把无外载编码特征拆分成镜像偶/奇分量；外力/力矩亦按镜像变换拆分为偶分量（Fx、Fz、My）和奇分量（Fy、Mx、Mz），通过镜像兼容的调制与跨支路耦合构建 actor。
- **Cross-load consistency**：以相同初态和速度命令并行运行空载、左载与右载的成对片段，约束窗口末端的平面位移与航向响应，使负载改变尽量不改变 commanded motion。
- 使用强化学习训练；文中说明混合匹配初态的负载组和常规随机环境，50 Hz 控制，4096 并行仿真环境。

## 实验与结论

- 仿真对 FALCON 比较中，直行任务的纵向速度误差 0.0632 对 0.1033、侧向 0.0824 对 0.1212、航向 0.0924 对 0.5115（按原文对应表项）。论文还在多个负载场景报告一致性与执行器负担改进。
- G1 真机固定导航设置：左右手各持 **1.5 kg** 负载，在两个环境组合进行合计 **40 次**测试；ReACT **40/40** 次无碰撞到达目标，而 FALCON **0/40** 次完成该测试。
- **反例必须保留：** 真机转弯时 ReACT 侧向速度 MAE **0.159**，FALCON **0.135**，不能称其对所有真机指标绝对占优。
- 以上为作者报告的受控实验，不代表不同机器人、未知货物几何、室外路况或自发运动操作的普遍保证。

## 开源状态

论文链接已核对；截至 2026-11-12 本次一手资料中**未找到可验证的作者官方实现与可运行入口**，故不得虚构源码时序图。后续若官方公布仓库，可追加源码核对。

## 对 wiki 的映射

- [ReACT 实体页](../../wiki/entities/paper-react-asymmetric-load-tracking.md)
- [人形机器人移动](../../wiki/tasks/humanoid-locomotion.md)
- [运控模型评测](../../wiki/concepts/motion-control-policy-evaluation-metrics.md)

## 参考来源

- [arXiv v1 全文](https://arxiv.org/html/2611.13929v1)
- [arXiv 论文记录](https://arxiv.org/abs/2611.13929)
