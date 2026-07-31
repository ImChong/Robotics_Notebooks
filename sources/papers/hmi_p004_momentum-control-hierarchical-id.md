# Momentum Control with Hierarchical Inverse Dynamics on a Torque-Controlled Humanoid（Momentum Control，HMI P004）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Momentum Control with Hierarchical Inverse Dynamics on a Torque-Controlled Humanoid
- **短名：** Momentum Control
- **类型：** paper / hmi-papers / 工程与实机部署
- **HMI ID：** P004
- **年份：** 2016
- **原文：** 见 HMI 论文总索引
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 以质心线/角动量为高层平衡目标，用层级逆动力学在浮基动力学与接触约束内求加速度、接触力与力矩。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P004](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P004.md)

## 开源状态（步骤 2.5）

- **结论：** 未作为本库主复现入口核验；以原文为准

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

力矩控制人形的难点不是算出一个好看的关节轨迹，而是同时满足浮基动力学、接触约束和任务优先级。本文把质心线动量与全身角动量作为高层平衡目标，再用层级逆动力学求关节加速度、接触力和力矩，让“机器人整体怎样运动”先于局部关节姿态。

**对 wiki 的映射：** [`wiki/entities/paper-momentum-control-hierarchical-id.md`](../../wiki/entities/paper-momentum-control-hierarchical-id.md)

### 摘录 2

控制器根据估计状态、接触集合和任务参考建立一组约束。浮基刚体动力学保证求得的加速度、接触力与关节力矩彼此一致；接触加速度约束让支撑点不随意滑动；摩擦和单边接触限制接触力；力矩与关节约束限制硬件可执行范围。在这些硬约束内，第一层跟踪期望质心/角动量变化，后续层再处理摆动脚、躯干、手和姿态。

**对 wiki 的映射：** [`wiki/entities/paper-momentum-control-hierarchical-id.md`](../../wiki/entities/paper-momentum-control-hierarchical-id.md)

### 摘录 3

决策量不是一条抽象“全身动作”，而是广义加速度、接触力与关节力矩的动力学一致组合。输入来自状态估计器、接触计划和各任务参考，输出最终力矩给真实执行器。动量目标通常由质心/姿态误差和期望外力形成，摆动脚与手等局部任务只能在高优先级动力学和接触可行域内优化。

**对 wiki 的映射：** [`wiki/entities/paper-momentum-control-hierarchical-id.md`](../../wiki/entities/paper-momentum-control-hierarchical-id.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-momentum-control-hierarchical-id.md`](../../wiki/entities/paper-momentum-control-hierarchical-id.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
