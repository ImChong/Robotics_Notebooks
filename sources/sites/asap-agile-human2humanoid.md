# ASAP 项目页（agile.human2humanoid.com）

> 来源归档（site / project page）

- **标题：** ASAP — Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills
- **类型：** project site
- **URL：** <https://agile.human2humanoid.com/>
- **论文：** <https://arxiv.org/abs/2502.01143>（RSS 2025）
- **代码：** <https://github.com/LeCAR-Lab/ASAP>（MIT，已开源）
- **框架依赖：** <https://github.com/LeCAR-Lab/HumanoidVerse>
- **机构：** 卡内基梅隆大学（CMU）；英伟达（NVIDIA）
- **核查日期：** 2026-09-22
- **一句话说明：** LeCAR-Lab 敏捷人形全身技能 Sim2Real 项目页：两阶段 delta action 对齐仿真与真机动力学，展示 Unitree G1 侧跳/前跳/球星动作等 demo，并链到官方 ASAP 代码与数据集。

## 核心摘录（归纳，非全文）

- **问题：** 敏捷全身动作受 sim–real 动力学失配制约；SysID 与 DR 往往要么调参昂贵，要么策略保守牺牲敏捷性。
- **方法（四步）：** (1) 人类视频重定向参考 → 仿真预训练 motion tracking；(2) 真机 rollout 收集轨迹 → 训练 delta action 模型对齐 \(s_t\) 与 \(s^r_t\)；(3) 冻结 delta 模型嵌入仿真器 → 微调预训练策略；(4) 真机部署时 **去掉** delta 模型，直接运行微调策略。
- **评测场景：** IsaacGym→IsaacSim、IsaacGym→Genesis、IsaacGym→真机 Unitree G1；相对 SysID、DR、仅学 delta 动力学不回灌等基线降低跟踪误差。
- **Demo：** 项目页含 LeBron James 等球星动作 Before/After ASAP 对比 GIF。

## 开源状态

- 项目页未单独列 Hugging Face，但 **GitHub 代码已发布**（MIT）：motion tracking、delta action 训练、AMASS 重定向、MuJoCo sim2sim、UnitreeSDK sim2real、ASAP motion 数据集均已标记完成（README TODO 全勾）。
- **已开源**；详见 [`sources/repos/asap.md`](../repos/asap.md) 与 [`sources/repos/humanoidverse.md`](../repos/humanoidverse.md)。

## 对 wiki 的映射

- [paper-notebook-asap-aligning-simulation-and-real-world-physics](../../wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md)
- [paper-hrl-stack-25-asap](../../wiki/entities/paper-hrl-stack-25-asap.md)
- [HumanoidVerse 框架](../../wiki/entities/humanoidverse.md)
- [human2humanoid](../../wiki/entities/human2humanoid.md)

## 参考来源（原始）

- 项目页：<https://agile.human2humanoid.com/>
- arXiv：<https://arxiv.org/abs/2502.01143>
- 代码：<https://github.com/LeCAR-Lab/ASAP>
