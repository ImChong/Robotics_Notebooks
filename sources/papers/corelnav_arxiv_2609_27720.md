# CoRelNav: Collaborative Relational Navigation for Multi-Robot Spatially Constrained Semantic Navigation

- **作者：** Jinyu He、Zihao Mao、Haonan Jin、Mengyin Fu、Wenjie Song；前两位共同第一作者
- **机构：** 北京理工大学自动化学院（School of Automation, Beijing Institute of Technology）
- **版本：** arXiv:2609.27720v1，2026-09-23；页面注明投稿 ICRA 2027，不代表录用
- **原文：** [arXiv](https://arxiv.org/abs/2609.27720) · [HTML](https://arxiv.org/html/2609.27720v1) · [8 页 PDF](https://arxiv.org/pdf/2609.27720v1)
- **核对范围：** PDF §I–V，方法公式 (1)–(7)、图 1–6 的流程与图注、表 I–III；交叉核对 arXiv HTML。未复现仿真或真机。
- **通读：** 已完成

## 原文核对笔记

- §III-C/D：共享占据栅格、实例记录和带观测的拓扑图各有分工。对象与全景的语义相关性用于筛探索前沿，团队最终分配还考虑预测覆盖的并集、补齐不确定候选证据的收益和运动代价。
- §III-E：验证假设绑定目标实例 ID。关系模型给整体和逐条子句的结果，规则门控检查支撑物 ID、类别、参数个数和证据。单节点拒绝只拒该次提议；不确定性可驱动补拍。跨节点用固定证据区域的并集裁剪最新点缓存，而非拼接几张历史图。
- §IV-A：混合任务整体 SR/SPL 为 0.513/0.129，VLN-Game (Dual) 为 0.395/0.084；每回合模型用量为 91.2k 对 189.5k tokens。比较不隔离单一组件，外部基线的子任务按顺序发放；消融另见表 III。
- §IV-B：两台轮式机器人在未见环境做端到端演示。论文展示一项代表性执行，未给真机批量成功率；仿真掩码和实例 ID 来自模拟器，真机感知误差未被系统量化。
- **公开材料：** 截至 2026-11-26，arXiv 摘要页、全文及按题名和作者检索未确认官方独立项目页、CoRelNav 实现或该论文的评测任务包。论文 TeX、引用的 HM3D、检测器和分割模型不等同于本项目源码或数据发布。

- **Wiki：** [CoRelNav](../../wiki/entities/paper-agentic-2609-27720.md)
