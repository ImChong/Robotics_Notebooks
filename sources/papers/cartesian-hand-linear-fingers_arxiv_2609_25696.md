# Cartesian Hand（arXiv:2609.25696）

> 来源归档（paper）

- **标题：** The Cartesian Hand: In-Hand Manipulation with All-Linear Fingers
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.25696>
- **PDF：** <https://arxiv.org/pdf/2609.25696>
- **项目页：** <https://generalroboticslab.com/cartesian_handv1>
- **代码：** <https://github.com/generalroboticslab/Cartesian_Hand>（**已开源**，Apache-2.0，2026-09-28 项目页 + GitHub 复核）
- **入库日期：** 2026-09-28
- **最近复核：** 2026-09-28（GRL 项目页 + `Cartesian_Hand` 仓库）
- **一句话说明：** 7-DoF 全线性末端：双独立平行夹爪 + 四平移指尖；35 类铰接物体手内操作；控制栈与仿真模型已开源。

## 开源状态

- **已开源（控制与仿真）**：`generalroboticslab/Cartesian_Hand` — 任务、Policy/TaskRunner、studio/sim/warp 后端、MuJoCo XML、STEP 源文件。
- **边界**：README 称 `cartesian_hand/tasks/` 为转录任务，**尚未在全部物体上重验证**；论文「全部软硬件设计开源」以仓库持续更新为准。

## 核心摘录

1. **机构：** Duke University，General Robotics Lab（Boyuan Chen 等；† 共一 Xia/Li）。
2. **机制：** 7 个棱柱关节 — $q_0,q_4$ 上下夹爪开闭；$q_1,q_2,q_5,q_6$ 指尖平移；$q_3$ 上下夹爪间距；配置无关的指尖运动学 → 线性运动原语组合。
3. **硬件（论文）：** Feetech STS3915 + 齿条齿轮；约 850 g；整手约 $500；PLA 结构 ~2 h 组装。
4. **评测：** 35 物体（实验室/制造/家用）；Franka Panda → 人形迁移；双臂各一只手做 bimanual 实验室操作。
5. **控制：** 关节反馈接触检测，无视觉信号要求（论文设定）。

**对 wiki 的映射**

- [paper-cartesian-hand-linear-fingers](../../wiki/entities/paper-cartesian-hand-linear-fingers.md)
- [cartesian-hand-v1-generalroboticslab.md](../sites/cartesian-hand-v1-generalroboticslab.md)
- [cartesian_hand.md](../repos/cartesian_hand.md)
