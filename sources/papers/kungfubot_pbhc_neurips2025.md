# KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills

- **标题：** KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills
- **类型：** paper
- **会议：** NeurIPS 2025
- **arXiv：** <https://arxiv.org/abs/2506.12851>
- **OpenReview：** <https://openreview.net/forum?id=LCPoXt0pzm>
- **项目页：** <https://kungfubot.github.io/>
- **代码：** <https://github.com/TeleHuman/PBHC>
- **机构：** TeleAI、SJTU、ECUST、HIT、ShanghaiTech
- **收录日期：** 2026-07-09
- **一句话说明：** 面向高动态人类行为（武术、舞蹈）的 physics-based 全身控制：多阶段 motion 处理（提取/滤波/校正/重定向）+ **双层优化自适应跟踪容差课程** + **非对称 actor-critic** RL，在 Unitree G1 真机稳定复现踢腿、旋子、太极等。

## 核心摘录（策展，非全文）

### 1）问题与动机

- 现有模仿学习多只能跟踪 **平滑、低速** 人类动作，即便精心设计 reward 与课程仍难覆盖武术级高动态。
- 目标：通过 **物理约束下的 motion 处理** 与 **自适应 motion tracking**，掌握 Kungfu、舞蹈等高难技能。

### 2）Motion 处理管线

- **提取 → 滤除 → 校正 → 重定向**，尽可能满足物理约束（支撑、无脚滑等）。
- 开源实现（PBHC）：视频/LAFAN/AMASS → SMPL → Mink 或 PHC → G1 机器人轨迹；依赖 GVHMR（视频）、IPMAN（滤波）等。

### 3）自适应 Motion Tracking（核心算法）

- 将跟踪精度容差建模为 **双层优化**：根据当前跟踪误差 **动态调整** 可接受偏差，形成 **自适应课程**。
- 对比固定 tracking factor：自适应机制在各动作上接近最优；固定因子对不同动作表现不稳定。
- 训练采用 **非对称 actor-critic**（部署侧 actor 仅用 onboard 观测）。

### 4）实验与真机

- 仿真：PBHC 在各难度级别 **一致优于** 可部署基线，接近 oracle（MaskedMimic 除外）。
- 真机（G1）：Jump kick、Roundhouse/Side/Front/Back kick、360° spin、舞蹈、太极、马步、组合拳等；sim-to-real 太极根轨迹与仿真对齐（真机根位置不可测，固定原点对比）。

## 对 wiki 的映射

- [paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont](../../wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md) — 主实体页
- [pbhc.md](../repos/pbhc.md) — 官方代码与模块说明
- [curriculum-learning](../../wiki/concepts/curriculum-learning.md) — 自适应跟踪容差课程
- [motion-retargeting](../../wiki/concepts/motion-retargeting.md) — SMPL→G1 重定向
- [paper-kungfuathlete-humanoid-martial-arts-tracking](../../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md) — 武术 tracking 姊妹

## 参考来源（原始）

- 项目页：<https://kungfubot.github.io/>
- 论文：<https://arxiv.org/abs/2506.12851>
- 代码：<https://github.com/TeleHuman/PBHC>
