# PyBullet 官方网站 (pybullet.org)

> 来源归档

- **标题：** Bullet Real-Time Physics Simulation — Home of Bullet and PyBullet
- **类型：** site（官方站点 / 论坛入口）
- **来源：** Erwin Coumans 维护
- **链接：** https://pybullet.org/wordpress/
- **入库日期：** 2026-06-17
- **一句话说明：** Bullet / PyBullet **官方主页与社区入口**：机器人 RL、模仿学习、Assistive Gym、Habitat 集成等案例索引；Colab 一键 `pip install pybullet`；链到 [bullet3](https://github.com/bulletphysics/bullet3) 发布与 PyBullet Quickstart。
- **沉淀到 wiki：** [PyBullet](../../wiki/entities/pybullet.md)

---

## 站点要点（机器人 / RL 向）

### 快速上手

- **Colab**：`!pip install pybullet` 约 15 秒（预编译 wheel）；可接 Stable Baselines PPO 训练 Gym 环境。
- **Quickstart Guide**（Google Doc，README 亦引用）：URDF 加载、`stepSimulation`、电机控制、传感器与 VR 共享内存模式。
- **论坛**：GitHub Issues 曾充斥支持帖已关闭；社区讨论以 [pybullet.org 论坛](http://pybullet.org) 为主。

### 代表性研究 / 生态（站点归档）

| 方向 | 代表工作 | 与本库交叉 |
|------|----------|------------|
| 四足模仿动物 | RSS 2020 Best Paper；[motion_imitation](https://github.com/google-research/motion_imitation) | [motion-imitation-quadruped](../../wiki/entities/motion-imitation-quadruped.md) |
| 可微仿真 | **TDS**（Tiny Differentiable Simulator）；NeuralSim (ICRA 2021) | 与 MuJoCo MJX / 可微物理对照 |
| 辅助机器人 | **Assistive Gym**（ICRA 2020）— 人机协作与偏好学习 | 操作 / HRI 仿真入口 |
| 具身导航 | **Habitat-Sim** 集成 Bullet Physics | 与 Isaac / MuJoCo 导航栈对照 |
| 合成数据 | **Kubric** + PyBullet + Blender | 视觉标注数据集生成 |
| 元学习 sim2real | 论文 arXiv:2003.01239 | [sim2real](../../wiki/concepts/sim2real.md) |

### 版本与发布

- 站点提及 **Bullet / PyBullet 3.05** 等 tagged release（含 FEM 可变形体改进）；以 [bullet3 releases](https://github.com/bulletphysics/bullet3/releases) 为准。

## 对 wiki 的映射

- [PyBullet（实体页）](../../wiki/entities/pybullet.md)
- [具身 RL 最小闭环](../../wiki/concepts/embodied-rl-minimal-closed-loop.md)
- [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
