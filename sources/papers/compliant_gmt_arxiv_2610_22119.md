# Compliant GMT（arXiv:2610.22119）

- **标题：** Compliant GMT: Making General Motion Tracking Compliant via Residual Reference Adaptation
- **作者：** Venki Parsuram、Michelle Wang、Jin-Hyuk Hong、Pulkit Agrawal
- **论文：** <https://arxiv.org/abs/2610.22119>（2026-10-15）
- **项目页：** <https://venki1899.github.io/c-gmt/>；[项目页归档](../sites/compliant-gmt.md)
- **入库日期：** 2026-10-27

## 摘要要点

- 通用运动跟踪器擅长多样运动，但刚性跟踪可能使机器人抵抗外力。
- 方法保留预训练运动跟踪器，以本体感知历史条件化的残差适配器调整参考动作，实现全身柔顺响应。
- 特权教师用真实接触力与指定刚度求解柔顺参考；学生从本体感知学习修正，并送入冻结的跟踪器。
- 作者报告仿真与 Unitree G1 真机实验中的跟踪、柔顺、安全性及鲁棒性结果。项目页展示遥操作、舞蹈及接触操作。
- 项目页标注代码即将发布；截至入库日未提供可运行代码链接。

## 对 wiki 的映射

- [Compliant GMT 实体](../../wiki/entities/paper-compliant-gmt.md)
