# AgenticGenTAMP — 作者官方项目页核查

> 原始来源归档（site），2026-11-10 核对。

- **标题：** Coding Agents for Generalized Task and Motion Planning Problems
- **主页：** <https://agenticgentamp.github.io/>
- **论文：** <https://arxiv.org/abs/2609.30233>
- **代码：** <https://github.com/tomsilver/robocode>
- **数据集/结果：** 作者项目页有 Dataset 和 Download results 链接；具体分发、复现实验种子说明见官方 README。
- **源码开放程度：** **已开源**（仓库提供代码/配置/运行说明，MIT）；不是只有宣传页。
- **核对范围：** 访问项目页 Title、Authors、Setup、Generalized TAMP、Access restrictions、Held-out evaluation、Results、Gallery/Video 与 Code/Dataset 链接；查看 README 对应复现协议。
- **通读：** 已完成
- **论文归档：** [AgenticGenPlan 论文](../papers/agenticgenplan_arxiv_2609_30233.md)
- **仓库归档：** [RoboCode](../repos/tomsilver-robocode.md)
- **对应 wiki：** [AgenticGenPlan](../../wiki/entities/paper-agenticgenplan-generalized-tamp.md)

## 重要文字限定

官网明确：28 个为不同实验环境，**每环境合成一个可复用策略**。仅对同一环境初始状态分布中的不同对象数量、布局、几何进行测试；不是统一策略控制全部 28 种环境。测试时程序冻结，无 LLM 在线调用。
