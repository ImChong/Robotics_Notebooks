# HLO-AMP（arXiv:2611.05113）

> 来源归档（paper）

- **标题：** HLO-AMP: Hierarchical Loco-Manipulation from Object-Aware Motion Priors
- **论文：** <https://arxiv.org/html/2611.05113v1>（v1，2026-11-03）
- **机构：** Robotics and AI Institute
- **入库日期：** 2026-11-05
- **核对范围：** 论文引言、方法、实验与部分附录；其余附录尚待逐项核对。
- **代码：** 论文未提供官方代码入口；尚未核实是否存在可运行实现。
- **映射：** [HLO-AMP](../../wiki/entities/paper-hlo-amp.md)

## 论文要点

- 用含物体的仿真场景和重定向 MoCap 训练低层运动模仿策略，再冻结供高层调用。
- 高层的命令限制在低层数据覆盖范围内；阶段调度用于训练，高层部署时不接收阶段标签。
- G1 的箱体状态来自外部动捕；20 秒仿真对比中整体成功率为 80.7%。
