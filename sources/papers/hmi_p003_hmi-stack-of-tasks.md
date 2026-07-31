# A Versatile Generalized Inverted Kinematics Implementation for Collaborative Working Humanoid Robots: The Stack of Tasks（Stack of Tasks，HMI P003）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** A Versatile Generalized Inverted Kinematics Implementation for Collaborative Working Humanoid Robots: The Stack of Tasks
- **短名：** Stack of Tasks
- **类型：** paper / hmi-papers / 工程与实机部署
- **HMI ID：** P003
- **年份：** 2009
- **原文：** 见 HMI 论文总索引
- **代码：** https://github.com/stack-of-tasks/sot-core
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 把任务、约束、雅可比与求解器组织成可动态插拔的软件任务栈，用零空间递归完成人形广义逆运动学。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P003](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P003.md)

## 开源状态（步骤 2.5）

- **结论：** 社区实现长期存在（stack-of-tasks 生态）；以官方/社区仓为准

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

很多人第一次接触Stack of Tasks，会把它理解成“按顺序解几个逆运动学”。这篇论文更重要的地方，是把任务、约束、雅可比、数值求解和机器人状态组织成可动态组合的软件实体，让复杂人形行为不必写成一个不可维护的大控制器。

**对 wiki 的映射：** [`wiki/entities/paper-hmi-stack-of-tasks.md`](../../wiki/entities/paper-hmi-stack-of-tasks.md)

### 摘录 2

每个任务提供当前误差、期望变化和对应雅可比。最高优先级任务先求一个关节速度解；下一任务只在前一任务雅可比的零空间里修正，依次向下堆叠。若低优先级目标与高优先级目标冲突，它只能完成可兼容的部分。关节限位、可视性、避碰等约束可以通过任务激活和不等式处理加入栈中，任务也能随行为阶段插入、删除或改变优先级。

**对 wiki 的映射：** [`wiki/entities/paper-hmi-stack-of-tasks.md`](../../wiki/entities/paper-hmi-stack-of-tasks.md)

### 摘录 3

求解器输入是当前广义位置、任务参考和各feature计算出的几何量，输出是广义速度或其积分后的姿态命令。任务对象将误差、雅可比、增益和激活条件封装在一起，solver只处理层级组合；这一软件边界使“看目标”“双手抓取”“保持质心”可以独立测试。它没有读取物体力或执行器状态，因此若接触任务需要力控，还要接操作空间/逆动力学层。

**对 wiki 的映射：** [`wiki/entities/paper-hmi-stack-of-tasks.md`](../../wiki/entities/paper-hmi-stack-of-tasks.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-hmi-stack-of-tasks.md`](../../wiki/entities/paper-hmi-stack-of-tasks.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
