# ZEST: Zero-shot Embodied Skill Transfer for Athletic Robot Control

- **标题**: ZEST: Zero-shot Embodied Skill Transfer for Athletic Robot Control
- **链接**: https://arxiv.org/abs/2602.00401
- **作者**: Jean-Pierre Sleiman, He Li, Alphonsus Adu-Bredu, Scott Kuindersma, Farbod Farshidian
- **机构**: Boston Dynamics, RAI Institute, ETH Zurich
- **发表日期**: 2026-02
- **核心关注点**: 跨形态技能迁移、高动态运动控制、多接触行为学习、零样本硬件部署

## 核心摘要

ZEST 提出了一种统一的框架，旨在通过强化学习（RL）将多样化的、异构的人类运动数据（MoCap 动捕、单目视频 ViCap、甚至是物理不真实的动画）直接转化为机器人鲁棒的高动态、多接触运动技能。

### 1. 核心挑战
- **多接触 (Multi-contact)**: 传统控制难以处理非足端接触（如爬行时的膝盖、肘部接触）。
- **技能工程 (Per-skill engineering)**: 避免为每项技能单独设计复杂的接触计划和参数。
- **灾难性遗忘 (Catastrophic Forgetting)**: 在长时程多样化数据集上训练时，模型容易遗忘早期的简单技能。

### 2. 技术贡献
- **自适应采样 (Adaptive Sampling)**: 基于失败率的指数移动平均（EMA）来衡量训练片段的难度，动态调整采样权重，确保资源集中在“难点”任务上。
- **虚拟辅助扳手 (Virtual Assistive Wrench)**: 在训练初期提供基座稳定辅助力，随训练进程自动衰减，实现了极高动态动作（如空翻）的自动课程学习。
- **残差动作空间 (Residual Actions)**: 策略输出关节目标的残差并叠加到参考轨迹上，提升了训练稳定性和效率。
- **极简 MDP**: 不依赖观测历史（History）或未来参考窗口，仅凭当前本体感知和下一步参考状态实现部署，降低了 Sim2Real 差距。

### 3. 实验验证
- **Atlas (全尺寸人形)**: 实现了侧手翻、连续后空翻、地板舞、战术爬行。
- **Unitree G1 (小型人形)**: 零样本迁移成功。
- **Spot (四足机器人)**: 证明了跨形态通用性。

## 对 Wiki 的映射
- **wiki/methods/zest.md** (新建)
- **wiki/concepts/sim2real.md** (补充高保真执行器建模与域随机化)
- **wiki/tasks/humanoid-locomotion.md** (补充多接触与高动态行为)
- **wiki/concepts/curriculum-learning.md** (补充虚拟辅助力辅助训练)
