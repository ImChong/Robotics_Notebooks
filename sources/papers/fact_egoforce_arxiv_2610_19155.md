# FACT: A Multimodal Force–Action–Track Foundation Model for Human-to-Robot Dexterity（arXiv:2610.19155）

> 来源归档（ingest）

- **标题：** FACT: A Multimodal Force–Action–Track Foundation Model for Human-to-Robot Dexterity
- **类型：** paper / dexterous-manipulation / tactile / human-to-robot-transfer / flow-matching
- **arXiv：** <https://arxiv.org/abs/2610.19155>（v1，2026-10-13）
- **HTML / PDF：** <https://arxiv.org/html/2610.19155v1> · <https://arxiv.org/pdf/2610.19155>
- **项目页：** <https://fact-1-0.github.io/> — 论文列出的作者网站；归档见 [项目页来源](../sites/fact-1-0-github-io.md)
- **代码与数据：** 截至 2026-10-16，未从论文页确认公开代码、数据或权重入口；项目页正文未能核实
- **机构：** 密歇根大学（University of Michigan）、斯坦福大学（Stanford University）、微软研究院（Microsoft Research）
- **作者：** Zhi Cao、Yining Hong、Jiaolong Yang、Yejin Choi、Li Fei-Fei、Jiajun Wu、Huang Huang
- **入库日期：** 2026-10-16
- **一句话说明：** 使用 EgoForce 人类触觉示范预训练；视觉轨迹引导粗动作规划，再联合预测未来力并精修动作，迁移到真实双臂灵巧操作。

## 摘要级要点

- **问题：** 大规模人类操作视频通常没有接触力标注；机器人灵巧操作演示采集昂贵。
- **数据：** EgoForce 用头戴与腕部三视角、双手触觉手套和自动力校准，采集约 2,000 小时、40,000 个 episode、1,000 项任务、1,000 名演示者。
- **方法：** Track-Guided Coarse Planner 联合预测未来视频、视觉点轨迹和动作块；Force–Action Joint Refiner 使用当前触觉历史，联合预测未来指尖力并精修剩余动作。
- **评测：** 十项真实双臂任务，每项任务用 10 条机器人轨迹与 90 条人类示范微调。摘要报告相对 T-Rex 的任务级最高提升 60 个百分点。
- **边界：** 数据规模不代表数据已公开可下载；小样本实机结果不能直接外推为部署可靠性。

## 对 wiki 的映射

- [FACT（EgoForce · 力–动作联合精修）](../../wiki/entities/paper-fact-egoforce-human-dexterity.md)
- 同名缩写的另一项工作：[FACT（Failure-Aware Causal Training，arXiv:2608.10232）](../../wiki/entities/paper-fact.md)
- 对照阅读：[VT-WAM（视觉–触觉–动作联合模型）](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md)

## 核心摘录

1. EgoForce 将校准力信号与视觉、手部动作同步采集；消融显示缺少人类预训练力监督时，部分接触任务明显退化。
2. 粗规划器联合预测未来视频、点轨迹和动作；精修器共同去噪剩余动作和未来指尖力，未来力是预测目标而非控制参考。
3. 七项二值任务各做 10 次 rollout；乐器任务按正确音符数计分；注射器旋钮同时报告参与次数与参与试次的读数误差。
