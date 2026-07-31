# Scalable and General Whole-Body Control for Cross-Humanoid Locomotion（XHugWBC，HMI P037）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Scalable and General Whole-Body Control for Cross-Humanoid Locomotion
- **短名：** XHugWBC
- **类型：** paper / hmi-papers / 动作跟踪与全身控制
- **HMI ID：** P037
- **年份：** 2026
- **原文：** https://arxiv.org/abs/2602.05791
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 用物理一致的随机形态扩展训练分布，并以语义关节映射与本体图网络对齐异构人形，检验不更新权重的跨人形控制边界。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P037](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P037.md)

## 开源状态（步骤 2.5）

- **结论：** 截至入库日未见稳定公开训练仓（以项目/论文页再核）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

普通RL策略的观测和动作按某台机器人固定关节顺序展开，换本体后维度、语义和动力学都变了。XHugWBC从训练分布、表示和网络三处同时处理：生成物理一致的随机形态，把机器人关节映射到全局语义空间，再用显式建模形态结构的策略学习一个跨人形控制器。

**对 wiki 的映射：** [`wiki/entities/paper-xhugwbc-cross-humanoid.md`](../../wiki/entities/paper-xhugwbc-cross-humanoid.md)

### 摘录 2

XHugWBC先从一套模板人形生成训练本体。形态随机化不会只改变腿长或躯干比例，而是同步调整几何、质量、惯量和关节参数，使每个虚拟机器人仍然对应物理一致的刚体系统。这样，策略训练时看到的不是一批外形不同但动力学相互矛盾的模型，而是覆盖肢段比例、质量分布和关节能力变化的本体族。跨本体策略能够适应新机器人，首先依赖这一训练分布提供足够的动力学变化。

**对 wiki 的映射：** [`wiki/entities/paper-xhugwbc-cross-humanoid.md`](../../wiki/entities/paper-xhugwbc-cross-humanoid.md)

### 摘录 3

不同机器人随后进入统一的32槽关节语义空间。每个髋、膝、腰、肩等关节按身体含义放到固定槽位，而不是沿用各自URDF中的名字和索引；目标机器人不存在或不可控的关节由可控性标记屏蔽。映射后的观测包含最近五步根角速度、投影重力、统一关节位置与速度和上一时刻动作，再拼接当前机器人可控制哪些关节以及全身运动命令。缺失关节使用mask而不是填成普通零值，是为了避免网络把“不存在的手腕自由度”误解为“存在但目标角度为零”。

**对 wiki 的映射：** [`wiki/entities/paper-xhugwbc-cross-humanoid.md`](../../wiki/entities/paper-xhugwbc-cross-humanoid.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-xhugwbc-cross-humanoid.md`](../../wiki/entities/paper-xhugwbc-cross-humanoid.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
