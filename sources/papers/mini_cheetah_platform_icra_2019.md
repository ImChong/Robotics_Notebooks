# Mini Cheetah: A Platform for Pushing the Limits of Dynamic Quadruped Control

> 来源归档

- **标题：** Mini Cheetah: A Platform for Pushing the Limits of Dynamic Quadruped Control
- **类型：** paper
- **作者：** Benjamin Katz, Jared Di Carlo, Sangbae Kim
- **机构：** MIT Biomimetic Robotics Lab
- **链接：** https://ieeexplore.ieee.org/abstract/document/8793865/
- **DOI：** https://doi.org/10.1109/ICRA.2019.8793865
- **会议：** ICRA 2019
- **代码：** https://github.com/mit-biomimetics/Cheetah-Software（控制软件栈）；硬件/驱动见 https://github.com/bgkatz
- **入库日期：** 2026-07-25
- **一句话说明：** 介绍 MIT Mini Cheetah 平台：~0.3 m / 9 kg、可背驱模块化执行器、Convex MPC 多步态至 2.45 m/s，以及离线非线性优化生成的 360° 后空翻。
- **开源状态：** **部分开源** — 控制软件 Cheetah-Software 已开源；执行器/驱动设计分散在 bgkatz 仓库与硕士论文。
- **沉淀到 wiki：** [paper-mini-cheetah-platform](../../wiki/entities/paper-mini-cheetah-platform.md)、[mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)

---

## 核心贡献（摘录）

1. **平台定位：** 小型、相对低成本但功率密度与抗冲击足够，便于单人操作与快速迭代控制算法。
2. **执行器：** 定制可背驱模块化执行器，支撑高带宽力控与撞击鲁棒。
3. **运动演示：** Convex MPC 实现 trot / trot-run / bounding / pronking，速度至 **2.45 m/s**；离线轨迹优化实现 **360° 后空翻**。

## 对 wiki 的映射

- [paper-mini-cheetah-platform](../../wiki/entities/paper-mini-cheetah-platform.md)
- [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)
- [benjamin-katz](../../wiki/entities/benjamin-katz.md)
- [model-predictive-control](../../wiki/methods/model-predictive-control.md)
