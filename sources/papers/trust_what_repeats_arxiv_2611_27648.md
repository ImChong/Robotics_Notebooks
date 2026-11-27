# Trust What Repeats: Capped World-Model Policy Gradients for Humanoid Motion Tracking · 原文核对

- **类型：** arXiv 预印本
- **作者：** Xuzhe Dang、Stefan Edelkamp
- **机构：** 捷克理工大学（Czech Technical University in Prague）、查理大学（Charles University）
- **版本：** v1，2026-11-20
- **原文：** [arXiv](https://arxiv.org/abs/2611.27648v1) · [HTML](https://arxiv.org/html/2611.27648v1) · [PDF](https://arxiv.org/pdf/2611.27648v1)
- **核对范围：** 通读 v1 的 13 页全文，核对动作导数定义、同起点复抽和仿真有限差分、两次反传、重要性采样及模型排序重置、表 I–VII 和附录 A–D；检索公开项目、代码与权重。
- **通读：** 已完成

| 位置 | 核对重点 |
|---|---|
| §III–IV，表 I | 裁剪单位是每条想象轨迹的动作梯度范数。复抽和沿同一方向做仿真有限差分是两种诊断；后者比的是回报变化，不是成功率。 |
| §V，附录 A–B | 先按采样率逆加权，再按加权范数限幅；仿真回合从片段重新开始的分布与想象起点抽样是两个机制。 |
| §VI，表 II–IV | G1 29-DoF 主试验为按动作划分的 BONES-SEED 留出集，三种子；23-DoF 扩展为每方法一粒种子；LAFAN1 的 40 段训练与评估重合。 |
| §VII，附录 D | 有限差分只在训练仿真中核验；阈值是经验值。§V-A 称每迭代的想象步数不变，但明列设置的名义乘积为 131,072 与 65,536；真实记账未核实。 |

截至 2026-11-27 未核验到本论文官方训练代码或检查点。[SuperMimic 官方仓库](https://github.com/HybridRobotics/supermimic) 是原基线项目，并非作者所用重实现。arXiv 提供的 TeX 源文件也不是训练源码。

- **沉淀到 wiki：** [Trust What Repeats](../../wiki/entities/paper-trust-what-repeats.md)
