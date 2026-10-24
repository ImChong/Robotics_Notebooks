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
