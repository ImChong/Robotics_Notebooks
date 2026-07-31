# A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation（OSF / Operational Space Formulation，HMI P001）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation
- **短名：** OSF / Operational Space Formulation
- **类型：** paper / hmi-papers / 工程与实机部署
- **HMI ID：** P001
- **年份：** 1987
- **原文：** https://doi.org/10.1109/JRA.1987.1087068
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 把运动与力控制直接写在末端任务空间动力学上，再用动态一致映射得到关节力矩，为后续任务优先级与全身控制奠基。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P001](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P001.md)

## 开源状态（步骤 2.5）

- **结论：** 不适用（经典论文，无可运行现代训练仓）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

这篇论文真正改变的不是一个控制增益，而是控制问题的表达方式。传统关节控制先给每个关节一个目标，再希望末端得到想要的运动；Khatib反过来问：如果任务本来就是“手沿某方向运动并在另一方向施力”，为什么不直接在末端任务空间定义动力学，再把结果映射成关节力矩？后来的操作空间控制、任务优先级和Whole-Body Control都继承了这个出发点。

**对 wiki 的映射：** [`wiki/entities/paper-operational-space-formulation.md`](../../wiki/entities/paper-operational-space-formulation.md)

### 摘录 2

机器人关节动力学给出质量矩阵、科氏/离心项、重力和关节力矩之间的关系，末端速度则由雅可比把关节速度映射到任务空间。关键不是简单使用雅可比转置，而是把关节空间惯量也一起映射过去，得到任务空间等效惯量 `Lambda`。这样，期望的任务空间加速度或力可以经过动态一致的映射变成关节力矩；控制器知道机器人沿不同方向“有多重”，而不是把所有方向当成同一种运动学误差。

**对 wiki 的映射：** [`wiki/entities/paper-operational-space-formulation.md`](../../wiki/entities/paper-operational-space-formulation.md)

### 摘录 3

冗余自由度也因此有了明确去处：主任务占用的方向由任务控制，剩余自由度通过动态一致零空间完成姿态、避限位等次任务，并尽量不扰动主任务。这里的“动态一致”很重要，普通伪逆只保证速度层面的投影，未必保证施加次任务力矩后主任务加速度不受影响。

**对 wiki 的映射：** [`wiki/entities/paper-operational-space-formulation.md`](../../wiki/entities/paper-operational-space-formulation.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-operational-space-formulation.md`](../../wiki/entities/paper-operational-space-formulation.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
