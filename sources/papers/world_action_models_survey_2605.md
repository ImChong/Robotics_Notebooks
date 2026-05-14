# World Action Models: The Next Frontier in Embodied AI（综述预印本）

- **类型**：论文（survey）
- **收录日期**：2026-05-14
- **arXiv**：<https://arxiv.org/abs/2605.12090>（Submitted 12 May 2026）
- **PDF**：<https://arxiv.org/pdf/2605.12090.pdf>
- **配套资源**：OpenMOSS [Awesome-WAM 仓库](../repos/awesome-wam-openmoss.md) · [静态站点](../sites/awesome-wam-openmoss.md)

## 一句话

提出并形式化 **World Action Models（WAM）**：把 **环境前向动力学预测** 与 **可执行动作生成** 放在同一具身策略框架里，目标分布为 **未来观测与动作的联合** \(p(o', a \mid o, l)\)，区别于纯反应式 VLA 的 \(p(a \mid o, l)\) 与单独世界模型的 \(p(o' \mid o, a)\)。

## 为什么值得保留

- 给出一组可操作的 **边界判据**：未来状态预测必须是策略的一部分，而不是外挂 backbone 或独立仿真器。
- 用 **Cascaded WAM**（先规划未来表征再解码动作）与 **Joint WAM**（共享模型联合预测未来与动作）两条主线整理文献，并延伸到生成模态、条件机制与动作解码策略。
- 系统梳理 **数据生态**（机器人遥操作、便携人类示教、仿真特权监督、互联网/自我中心视频）与 **评测**（视觉保真、物理常识、动作可推断性 + 机器人基准族）。

## 核心摘录（面向 wiki 编译）

### 与 VLA / 世界模型的关系（摘要口径）

- **VLA**：强语义接地，但多数仍是 **观测→动作** 的反应式映射。
- **World model**：刻画 \(p(o' \mid o, a)\)，偏预测而非直接可执行策略。
- **WAM**：联合 \(p(o', a \mid o, l)\)，要求 **前向预测与动作生成在训练与推理链路中耦合**。

### 架构分类（综述正文结构）

| 家族 | 直觉 | 主要张力 |
|------|------|----------|
| **Cascaded WAM** | `future plan → action`：世界模型先给出未来计划，再由动作头解码 | 两阶段 **对齐/信息瓶颈** |
| **Joint WAM** | `future + action`：共享表示下联合监督 | 模态、训练目标与 **推理延迟** |

子维度（站点与论文一致强调）：显式/隐式规划载体、自回归 vs 扩散/流匹配等生成方式、条件与解码策略。

### 数据与评测（索引级）

- **数据**：机器人对齐轨迹、UMI 类便携示教、仿真特权信号、大规模人/自我中心视频——综述强调 **混合设计** 而非单纯扩规模。
- **评测**：世界侧（PSNR/SSIM/LPIPS/FVD、物理常识基准族、WorldSimBench / IDM Turing Test 等）；策略侧（Meta-World、RLBench、LIBERO、RoboTwin、HumanoidBench、BEHAVIOR-1K 等按形态与任务族组织）。

### 开放挑战（综述列题）

架构耦合对比不足、多模态物理状态（触觉/力/形变）覆盖弱、数据混合边际收益不清、长程误差累积、扩散/自回归推理效率、缺少 **想象未来与真实执行因果一致** 的联合安全指标。

## 对 wiki 的映射

- 升格页面：[World Action Models（WAM）](../../wiki/concepts/world-action-models.md) — 概念定义、与 VLA/生成式世界模型的边界、架构族谱与阅读入口。
- 交叉补强：[VLA](../../wiki/methods/vla.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Being-H0.7](../../wiki/methods/being-h07.md)（潜空间世界–动作先验的先行实例）。
