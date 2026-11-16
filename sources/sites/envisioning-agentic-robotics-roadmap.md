# Envisioning Agentic Robotics — Grid Robotics 官方项目页

> 来源归档（site / official project page）

- **英文：** Envisioning Agentic Robotics: A New Era of Robotic Systems
- **中文：** 展望智能体机器人：机器人系统的新纪元
- **主页：** https://grid-robotics-official.github.io/agentic-robotics-roadmap/
- **论文：** https://arxiv.org/abs/2611.18359
- **论文归档：** [arXiv:2611.18359v1](../papers/envisioning_agentic_robotics_arxiv_2611_18359.md)
- **wiki：** [论文详情](../../wiki/entities/paper-envisioning-agentic-robotics.md)
- **核对日期：** 2026-11-16
- **核对范围：** 官方页的 Paradigm Shift、Reference Framework、Embodied System 2、Evaluation、Digital versus Physical Agents、Road Ahead、Applications、Conclusion 与页尾资源区；同 arXiv v1 交叉核对。
- **通读：** 已完成
- **代码：** **未发现本论文独立可运行代码链接**（截至核对日；页面是方法论/研究路线展示，并非新模型训练发布）。
- **数据和权重：** 未发现本论文独立发布的数据集或模型权重下载入口；页面引用其他工作的基准链接不等于它们是本文发布的资产。

## 关键信息

1. 框架使用 S2 决策验证、S1 技能、S0 机体实时控制和独立的事件监控；支持 layered orchestration、direct control、code as policy 三种调用路径。
2. 具身 harness 的核心是上下文、工具接口、规划、记忆与验证；评测覆盖 9 维、53 子维度，其中 29 个与具身相关。
3. 研究路线用 L1 primitive → L2 task → L3 workflow → L4 standing role 描述可委托职责；工作区（workcell、room、building、site）是另一条独立坐标轴。
4. 站内比较案例来自其引用的不同原始研究，条件与协议不同；不能合并为本文自己做的统一实验榜单。

## 开源与复现边界

官方页提供论文与相关文章引用，但核对日未列本研究可运行源码、模型权重或数据。**源码运行时序图不适用**，不可凭架构示意图编造 `train.py` 或 ROS 控制调用顺序。
