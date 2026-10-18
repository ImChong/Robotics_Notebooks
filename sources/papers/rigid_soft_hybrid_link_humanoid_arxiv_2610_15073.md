# Modeling and Control of a Rigid-soft Hybrid-link Humanoid Robot（arXiv:2610.15073）

> 来源归档（ingest）

- **标题：** Modeling and Control of a Rigid-soft Hybrid-link Humanoid Robot
- **类型：** paper / humanoid control / rigid-soft hybrid system
- **来源：** arXiv 预印本（2026-10-10）；期刊版本发表于 Robotics and Autonomous Systems, 203, 105515 (2026)
- **arXiv：** <https://arxiv.org/abs/2610.15073>
- **DOI：** <https://doi.org/10.1016/j.robot.2026.105515>
- **作者：** Zewen He、Taiki Ishigaki、Ko Yamamoto
- **机构：** 东京大学（The University of Tokyo）
- **入库日期：** 2026-10-18
- **一句话说明：** 将 PCS 柔性体模型与传统刚体多连杆系统结合，提出面向刚-软混合人形机器人的逆运动学与柔顺优化控制，并以正向动力学仿真和带板簧假肢腿的人形平台验证。

## 摘要级要点

- **问题：** 被动柔性构件能够储能并缓解动态运动冲击，但其连续变形、被动弹性与刚体关节控制相互耦合，使人形机器人的运动生成与平衡控制更复杂。
- **建模：** 用分段常应变模型（Piece-wise Constant Strain, PCS）表示柔性结构的连续变形，并与刚体多连杆模型集成成刚-软混合系统。
- **控制：** 提出考虑弹性回复力与重力平衡的逆运动学方法，以及考虑被动黏弹性的柔顺优化控制；优化对象包含主动关节柔顺矩阵，控制量采用力/力矩语义。
- **仿真验证：** 正向动力学仿真覆盖站立、垂直跳跃和双足行走。
- **实验验证：** 构建带板簧式假肢腿的双足人形实验平台，通过运动捕捉测量假肢变形；真机实验验证双支撑和单支撑下的站立平衡控制。
- **证据边界：** 论文报告的跳跃与行走结果来自仿真；实验部分验证的是站立平衡，不能将仿真动态动作表述为真机演示。

## 研究脉络（相关论文）

| 论文 | 关系 |
|------|------|
| [Compliance Optimization Control for Rigid-Soft Hybrid System and its Application in Humanoid Robot Motion Control](https://doi.org/10.1109/IROS58592.2024.10801838)（IROS 2024） | 同作者团队的前序工作，提出刚-软混合系统的柔顺优化控制与仿真；期刊论文进一步加入逆运动学、行走仿真和站立平衡实验。 |
| [Dynamics Computation of a Hybrid Multi-link Humanoid Robot Integrating Rigid and Soft Bodies](https://www.ynl.t.u-tokyo.ac.jp/wp/publications/publications-2021/)（IROS 2021） | 更早的刚-软混合人形动力学建模工作，为后续将 PCS 柔性体与浮动基刚体系统组合提供研究背景。 |

## 对 wiki 的映射

- 主实体：[刚-软混合链路人形机器人建模与控制](../../wiki/entities/paper-rigid-soft-hybrid-link-humanoid.md)
- 相关概念：[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Impedance Control](../../wiki/concepts/impedance-control.md)
- 来源身份：arXiv:2610.15073；期刊 DOI: 10.1016/j.robot.2026.105515；前序会议论文 DOI: 10.1109/IROS58592.2024.10801838

## 当前提炼状态

- [x] 论文摘要与实验边界已归档
- [x] wiki 页面映射确认
- [x] 相关页面参考来源与交叉链接已补充
- [ ] 论文未提供可运行代码入口；如后续出现官方项目页或代码仓库，再补充开源核查与复现路径
