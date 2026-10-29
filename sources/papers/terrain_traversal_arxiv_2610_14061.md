# Terrain Traversal Without Motion Priors: Expert-Routed Humanoid Locomotion at Deployment Fidelity

> 论文资料归档，非原文镜像。

- **类型：** arXiv 预印本
- **论文：** <https://arxiv.org/abs/2610.14061>
- **项目页：** <https://fai-rl-tech.github.io/moe-perceptive/>
- **作者：** Thanh Ly、Chien Le、Truong-Duy Dang、Duy M. H. Nguyen、Duy-Dung Le、Vien Anh Ngo、An T. Le
- **日期：** 2026-10-07
- **代码：** 截至 2026-10-27，项目页 Code 按钮未指向源码仓库；数据和权重未见公开链接。

## 摘要要点

1. 感知高程图驱动地形分类器，路由五个专家；无需动作捕捉重定向、生成式动作先验或蒸馏。
2. 基于模型的落足规划在插值地形代价场执行六步闭式梯度优化；72.4% 的落点跨越至少一个栅格边界。
3. 在第二个面向部署保真度的仿真器中，85 米五段路线完成 18/20；去掉专家混合与落足优化的联合消融在楼梯和栏杆段均为 0%。该实验不是实机测试。

## 对 wiki 的映射

- [论文实体](../../wiki/entities/paper-terrain-traversal-without-motion-priors.md) — 方法、评测和边界。
- [楼梯与障碍感知运动](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md) — 任务背景。
