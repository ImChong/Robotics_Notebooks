# Distributed Secure Learning Control for Large-scale Multirobots under Stealthy Actuator Attacks（arXiv:2609.06896）

> 来源归档（paper）

- **标题：** Distributed Secure Learning Control for Large-scale Multirobots under Stealthy Actuator Attacks
- **作者：** Xinglong Zhang, Qingwen Ma, Cong Li, Hui Yin, Changxin Zhang, Yueying Wang, Wei Pan, Xin Xu
- **机构：** 国防科技大学、湖南大学、上海大学、曼彻斯特大学
- **类型：** paper
- **发表信息：** IEEE Transactions on Robotics，Early Access，2026；arXiv 预印本于 2026-09-07 发布
- **arXiv：** <https://arxiv.org/abs/2609.06896>
- **PDF：** <https://arxiv.org/pdf/2609.06896>
- **IEEE Xplore：** <https://ieeexplore.ieee.org/iel8/8860/4359257/11710271.pdf>
- **入库日期：** 2026-10-14
- **一句话说明：** 将大规模多机器人分布式 MPC 写成对抗攻击者与防御者的有限时域零和微分博弈，并以分布式 attacker–actor–critic 在线学习显式闭环策略。

## 开源状态

- **未发现公开代码仓库或独立项目页**（截至 2026-10-14；核对 arXiv HTML 与论文入口）。论文提供 arXiv 全文与补充材料；这不等同于可运行代码发布，复现实现状态待后续核查。

## 核心摘录

1. **问题与任务：** 面向存在隐蔽执行器攻击的多机器人系统（MRS），统一覆盖一般编队、仿射编队与 containment 控制。
2. **方法：** 在分布式模型预测控制（DMPC）的每个预测区间内构建攻击者—防御者零和微分博弈；attacker–actor–critic 同时在线生成攻击策略与防御策略的显式闭环形式，以启发式前向策略迭代替代在线求解完整非线性优化问题。
3. **规模与实机验证：** 仿真扩展至 1000 台机器人；多台轮式机器人上验证三类协同任务。作者报告 100 台情形下 DSLC 平均计算时间约 10.1–10.2 ms，而对照 DRMPC 未能给出结果。
4. **边界：** 当前分析假设通信拓扑静态且没有通信时延，学习时已知攻击概率；实机验证限于轮式地面机器人。

**对 wiki 的映射**

- [paper-dslc-multirobot-secure-control](../../wiki/entities/paper-dslc-multirobot-secure-control.md)
