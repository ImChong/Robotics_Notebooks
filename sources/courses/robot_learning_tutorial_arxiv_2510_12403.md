# Robot Learning: A Tutorial（arXiv:2510.12403）

- **类型：** course / research tutorial
- **标题：** Robot Learning: A Tutorial
- **作者：** Francesco Capuano、Caroline Pascal、Adil Zouitine、Thomas Wolf、Michel Aractingi
- **原文：** https://arxiv.org/abs/2510.12403
- **HTML（v1）：** https://arxiv.org/html/2510.12403v1
- **PDF：** https://arxiv.org/pdf/2510.12403
- **代码与示例：** https://github.com/fracapuano/robot-learning-tutorial
- **提交日期：** 2025-10-14（arXiv v1）
- **一句话说明：** 以经典模型式机器人学的工程局限为起点，引出强化学习、模仿学习和通用策略，并提供 LeRobot 实践代码。

## 阅读路线

- **§2.1 Explicit and Implicit Models：** 对照显式的运动学 / 动力学模型与从传感器动作数据中学习的隐式模型。
- **§2.2 Different Types of Motion：** 区分 manipulation、locomotion 与 mobile manipulation。
- **§2.3 Planar Manipulation：** 用平面机械臂讲解运动学、逆运动学与差分逆运动学；§2.3.1 补充反馈控制回路。
- **§2.4 Limitations of Dynamics-based Robotics：** 归纳模块集成、感知扩展、接触与摩擦建模、跨任务复用及机器人数据规模化等挑战。
- **§3 开篇 Robot (Reinforcement) Learning：** 从 §2 的限制过渡到学习式感知到动作控制，并说明数据驱动方法与显式动力学模型的关系。

## Figure 8 与 Figure 9

- **Figure 8：** 教程对动力学式机器人系统局限的图示总结：多组件集成、按任务定制感知处理、简化物理模型，以及未利用增长中的机器人数据。图注注明摩擦示意图的来源。
- **Figure 9：** 教程对学习式机器人控制的概括：统一控制器直接处理高维传感器输入，以交互数据学习策略，并展示数据规模与效果的经验联系。
- 两图是教程中的概念综合图；具体机制和实验结论应回到下列引用原文核查。

## 教程引用的一手资料

| 来源 | 原始入口 | 用途 / 年份核对 |
|------|----------|----------------|
| Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. | [MIT Press](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) · [作者在线书页](http://incompleteideas.net/book/the-book-2nd.html) | RL 基础教材；2018 年第二版 |
| Kober, Bagnell & Peters, “Reinforcement Learning in Robotics: A Survey” | [期刊 DOI](https://doi.org/10.1177/0278364913495721) · [期刊页面](https://journals.sagepub.com/doi/full/10.1177/0278364913495721) | 机器人强化学习综述；2013 年 |
| O’Neill et al., “Open X-Embodiment: Robotic Learning Datasets and RT-X Models” | [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) · [ICRA 2024 PDF](https://www.research.ed.ac.uk/files/466209726/ONeillEtalICRA2024OpenX-Embodiment.pdf) | 跨本体数据集与 RT-X；教程参考文献写 2025，但论文于 2023 年提交 arXiv、发表于 ICRA 2024 |
| Khazatsky et al., “DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset” | [arXiv:2403.12945](https://arxiv.org/abs/2403.12945) · [项目主页 / 数据入口](https://droid-dataset.github.io/) | DROID 示范数据集；原论文于 2024 年提交，项目页记录 2025 年相机标定更新，教程参考文献记为 2025 |

## 与本库关系

本教程适合作为从经典控制与建模过渡到 robot learning 的导读。它是概念教程，不是单一算法论文；图 8/9 的概括应与原始论文和本库主题页配合阅读。

- **LeRobot 实体：** [wiki/entities/lerobot.md](../../wiki/entities/lerobot.md)
- **强化学习方法：** [wiki/methods/reinforcement-learning.md](../../wiki/methods/reinforcement-learning.md)
- **Sutton & Barto 教材：** [wiki/entities/sutton-barto-rl-book.md](../../wiki/entities/sutton-barto-rl-book.md)
- **Open X-Embodiment：** [wiki/entities/paper-open-x-embodiment.md](../../wiki/entities/paper-open-x-embodiment.md)
- **DROID：** [wiki/entities/paper-rcl-ref-aa9c28fbf4242ea03696-droid-a-large-scale-in-the-wild-robot-manipulati.md](../../wiki/entities/paper-rcl-ref-aa9c28fbf4242ea03696-droid-a-large-scale-in-the-wild-robot-manipulati.md)

## 2026-11-09 原文全文复核

- **核对范围：** 2026-11-09 通读 arXiv:2510.12403v1 主文（前言、§1–6）、LeRobotDataset/数据采集、RL/SAC/HIL示例、BC与VAE/扩散/流匹配、ACT/DP、异步推理与 π₀/SmolVLA 示例、全部图表说明及文末引用表；未逐一执行 LeRobot 代码。
- **通读：** 已完成

- 教程既有 §2 的 FK/IK、diff-IK 与反馈闭环，指出摩擦/接触建模与分布变化对显式管线的局限；作者并未主张用端到端学习完全替代经典控制。
- §3 以 MDP/RL、SAC 与真实 HIL 引出奖励设计/交互成本/Sim2Real 难题；§4 BC 用专家轨迹学习分布，分别介绍 VAE、DDPM 与 flow matching 的生成机制、ACT/DP 的动作块接口。
- §4.4 的异步推理把机器人客户端执行已有 action chunk 与远端策略服务器推理分离；队列低于阈值时触发非阻塞新观测推理，并做观测去重与 overlap 合并，不能直接等同每个控制周期都等待 GPU。
- §5 比较 VLM/VLA、π₀ 的 VLM 与 action expert、SmolVLA 的小模型及社区数据集；§6 讨论传统模型和学习策略互补。
- 注意：arXiv v1 明示 2025-10-14，而当前 HTML 正文自带 Date: August 24, 2026 字样；归档身份按 arXiv v1，不把页面内部日期当投稿时间。

> “全文通读”仅指本篇官方原文的主要章节及原文附录，不等同逐篇审计所有参考论文、运行作者代码或复现实验。
