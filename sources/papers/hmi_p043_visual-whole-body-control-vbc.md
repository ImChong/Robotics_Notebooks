# Visual Whole-Body Control for Legged Loco-Manipulation（VBC，HMI P043）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Visual Whole-Body Control for Legged Loco-Manipulation
- **短名：** VBC
- **类型：** paper / hmi-papers / LocoManip
- **HMI ID：** P043
- **年份：** 2024
- **原文：** https://arxiv.org/abs/2403.16967
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 特权高层先学任务目标再蒸馏为视觉策略，低层全身控制执行基座与手臂命令，明确感知规划与身体控制分工。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P043](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P043.md)

## 开源状态（步骤 2.5）

- **结论：** 部分开源线索以原文/项目页为准

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

让图像策略直接输出所有腿和臂关节，既难训练又难Sim2Real。VBC把任务拆成两个频率层：低层goal-reaching controller接收机身速度与末端位姿目标，负责稳定执行；高层视觉策略根据物体深度图不断更新这些短期命令。高层学“下一步去哪、手往哪伸”，低层学“身体怎样做到”。

**对 wiki 的映射：** [`wiki/entities/paper-visual-whole-body-control-vbc.md`](../../wiki/entities/paper-visual-whole-body-control-vbc.md)

### 摘录 2

低层命令包括末端位置与姿态、前向速度和偏航速度。RL策略读取基座、腿、臂、接触、上一动作、步态时序和环境latent，输出12个腿关节目标；机械臂目标则通过Jacobian伪逆IK转换成关节角。随机采样不同末端目标与移动速度后，腿会通过弯曲、移动和调整基座扩大手臂可达空间。这里的whole-body指系统协同，并不意味着一个策略直接输出全部19自由度。

**对 wiki 的映射：** [`wiki/entities/paper-visual-whole-body-control-vbc.md`](../../wiki/entities/paper-visual-whole-body-control-vbc.md)

### 摘录 3

任务teacher能访问物体点云特征、准确位姿和本体状态，用RL输出末端增量、移动速度与抓手开合。部署student只看物体mask、分割深度、本体和上一高层动作，通过DAgger在student访问到的状态上学习teacher纠正。相机位姿、深度噪声、高低层调用比率和机械臂PD参数都在训练中随机化。

**对 wiki 的映射：** [`wiki/entities/paper-visual-whole-body-control-vbc.md`](../../wiki/entities/paper-visual-whole-body-control-vbc.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-visual-whole-body-control-vbc.md`](../../wiki/entities/paper-visual-whole-body-control-vbc.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
