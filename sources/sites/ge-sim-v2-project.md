# GE-Sim 2.0 项目页

> 来源归档

- **标题：** Genie Envisioner World Simulator 2.0 — Project Page
- **类型：** site
- **URL：** <https://ge-sim-v2.github.io/>
- **入库日期：** 2026-05-30
- **一句话说明：** 官方演示页：统一动作控制信号注入、多视角一致生成、分钟级长视频、大规模真机数据泛化、视觉+本体联合未来预测、World Judge 自动任务评测、加速推理演示。
- **沉淀到 wiki：** [GE-Sim 2.0](../../wiki/entities/ge-sim-2.md)

---

## 页面能力要点（策展）

1. **Control-Signal Injection**：异构动作空间/策略标定到统一控制空间，像素对齐的 EE pose 条件。
2. **Multi-View Consistency**：头相机 + 左右手相机；盲区物体随臂运动在多视角间一致（含镜面反射案例）。
3. **Minute-Level Generation**：长视界策略的稳定视频 rollout。
4. **Large-Scale Real Data**：百万级真机 episode（遥操作、部署、交互）。
5. **Visual + Proprioceptive Prediction**：状态专家从视频 latent 解码关节/夹爪，雷达图对比条件 vs 生成状态。
6. **World Judge**：语言任务描述下对 rollout 自动评分（如「把蓝色筹码放进红盒」）。
7. **Efficient Simulation**：加速框架演示（与论文「25 帧 / 2.3s @ H100」口径一致，以技术报告为准）。

## 应用场景（页面列举）

- 世界模拟器内 **策略评测**
- 世界模拟器内 **遥操作**
- 世界模拟器内 **闭环学习**

## 对 wiki 的映射

- [GE-Sim 2.0](../../wiki/entities/ge-sim-2.md) — 演示能力与产品叙事
- [sources/papers/ge_sim_2_arxiv_2605_27491.md](../papers/ge_sim_2_arxiv_2605_27491.md) — 方法细节以论文为准
