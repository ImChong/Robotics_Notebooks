# Learning to Ball: Composing Policies for Long-Horizon Basketball Moves（arXiv:2509.22442）

> 来源归档（ingest）

- **标题：** Learning to Ball: Composing Policies for Long-Horizon Basketball Moves
- **类型：** paper / physics-based-animation / hierarchical-rl / policy-composition / basketball
- **arXiv abs：** <https://arxiv.org/abs/2509.22442>
- **arXiv HTML：** <https://ar5iv.labs.arxiv.org/html/2509.22442>
- **PDF：** <https://arxiv.org/pdf/2509.22442>
- **Venue：** ACM TOG / SIGGRAPH Asia 2025（Vol. 44, No. 6，DOI [10.1145/3763367](https://doi.org/10.1145/3763367)）
- **项目页：** <https://pei-xu.github.io/basketball> — 归档见 [`sources/sites/pei-xu-basketball-github-io.md`](../sites/pei-xu-basketball-github-io.md)
- **代码：** <https://github.com/xupei0610/basketball> — 归档见 [`sources/repos/learning-to-ball.md`](../repos/learning-to-ball.md)
- **机构：** Stanford University（The Movement Lab）、University of California, Riverside、Roblox、Clemson University
- **分类（Paper Notebooks）：** 13_Physics-Based_Animation
- **入库日期：** 2026-07-28
- **一句话说明：** 用 **策略组合框架** 把差异极大的篮球子技能拼成长程连招，并以 **高层 soft router** 处理目标不清晰的过渡段；Isaac Gym 仿真角色可实时响应用户指令完成 shoot-off-the-dribble 等，**不依赖球轨迹参考**。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://pei-xu.github.io/basketball> | 方法拆解视频、子技能与过渡演示、多人交互 |
| 代码仓库 | <https://github.com/xupei0610/basketball> | MIT；`main.py` 训练/评测；`pretrained/` 子技能权重 |
| 方法前作 | [Composite Motion Learning with Task Control](https://pei-xu.github.io/CompositeMotion) | 多目标复合运动学习底子（arXiv:2305.03286） |
| 模仿底子 | [ICCGAN](https://pei-xu.github.io/ICCGAN) | GAN-like 物理模仿（arXiv:2105.10066） |
| 适配工具 | [AdaptNet](https://arxiv.org/abs/2310.00239) | 策略潜空间适配（过渡训练中用） |
| 篮球统一模仿对照 | [SkillMimic](https://arxiv.org/abs/2408.15270) | 统一 HOI 模仿 + HLC 离散选技能 |
| Paper Notebooks 进度锚点 | [`humanoid_pnb_learning-to-ball.md`](./humanoid_pnb_learning-to-ball.md) | 姊妹仓库深读笔记溯源 |

## 摘要级要点

- **问题：** 长程多阶段任务（如篮球连招）由 **目标明确的子任务** 与 **目标不清的过渡子任务** 交错组成；MoE / skill chaining 在子策略状态分布几乎不相交、或起终点难定义时会崩。
- **方法：** (1) 独立训好目标明确的子技能；(2) 用前驱策略作初始态分布、后继策略的 **state value** 塑形过渡奖励，并可与后继策略 **同步适配**；(3) 训 **高层 soft router** 按实时指令软性加权组合子策略动作。
- **三类过渡：** A 直接执行 → B 相互适配 → C 需中间策略（如 dribble→gather→shoot）。
- **数据：** 异构非结构化运动（网络视频姿态估计、无手部全身 mocap、自采手部手套、LAFAN1 跑步等）；**不假设球轨迹**，也不要求全身/手部一一对应。
- **主结果：** shoot-off-the-dribble 接球率 **98.3%**、投篮命中率 **91.8%**（职业场地宽区域网格评测）；启发式硬切换约 86.0% / 67.4%；硬路由训练显著弱于 soft router。

## 核心摘录（面向 wiki 编译）

### 1) 过渡类型与 Type C 训练

| 类型 | 条件 | 做法 |
|------|------|------|
| A Direct Execution | 前后策略共享可接管状态 | 直接切换 |
| B Mutual Adaptation | 后继需适应前驱终态 | value 塑形 + AdaptNet 适配 |
| C Intermediate Policy | 如 dribble↔shoot 不相容 | 训 gather：初态←前驱 rollout；奖励含 \(\bar{V}_{\text{shoot}}\)；同步适配 shoot |

Gather 奖励（论文式 (1)）：违规 → \(-1\)；否则 \(r_{\text{pose}} + 0.25\,\mathrm{Clip}(\bar{V}_{\text{shoot}}, -v, v)\)（\(v=1\)，PopArt 归一化）。

### 2) Soft router（式 (2)–(3)）

- 参考命令 \(\mathbf{c}_t\)（one-hot：运球 / gather / 投篮）+ 路由网络偏移 → 权重 \(\boldsymbol{\omega}_t\)，对子策略确定性动作做线性组合。
- 奖励鼓励 **某一子策略主导**（权重大于其余之和），同时允许过渡段轻微混合；相对 hard router（softmax one-hot）更易训、更稳。
- 训完后可 **蒸馏** 成单网络以降低推理开销。

### 3) 子技能与系统规模

- 约 **7** 类技能：dribble / shoot / layup 相关过渡 / run / turn / pick-up·rebound / defend 等；防守并入 locomotion。
- 仿真：Isaac Gym；角色 ~57 links / 76 DoF；策略 30 Hz、仿真 120 Hz；PPO + 对抗模仿（ICCGAN 族）。

### 4) 定量对照（shoot-off-the-dribble）

| 方法 | 接球率（量级） | 命中率（量级） |
|------|----------------|----------------|
| DirectExecution | ~0.7% | ~1.3% |
| NoAdapt | gather 高、接 shoot 后掉 | ~6.1% |
| SequentialChaining | 中等 | ~12.7% |
| 本文（含 soft router） | **98.3%** | **91.8%** |
| 无高层、仅启发式切换 | 86.0% | 67.4% |
| 预训练 shoot（参考态初始化） | — | 93.0%（上界参照） |

### 5) 开源核查（项目页 + GitHub，截至 2026-07-28）

| 组件 | 状态 |
|------|------|
| 训练/评测代码（`main.py` + `cfg/*.py`） | ✅ 已开源（MIT） |
| 子技能预训练（`pretrained/{dribble,shoot,catch,pass,rebound,locomotion+defend}`） | ✅ |
| 依赖说明（PyTorch 2.1.2 + Isaac Gym Pr4） | ✅ README |
| 高层 soft router / gather 独立 cfg 与预训练目录条目 | ⬜ 公开发布清单未列（完整长程组合需对照论文扩展训练） |

## 对 wiki 的映射

- 主实体页：[paper-notebook-learning-to-ball](../../wiki/entities/paper-notebook-learning-to-ball.md)
- 分类父节点：[paper-notebook-category-13-physics-based-animation](../../wiki/overview/paper-notebook-category-13-physics-based-animation.md)
- 方法背景：[Hierarchical RL](../../wiki/methods/hierarchical-reinforcement-learning.md)、[Imitation Learning](../../wiki/methods/imitation-learning.md)
- 前作索引：[Composite Motion Learning](../../wiki/entities/paper-notebook-composite-motion-learning-with-task-control.md)
- 篮球统一模仿对照：[SkillMimic](../../wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md)

## 参考来源（原始）

- Xu et al., *Learning to Ball: Composing Policies for Long-Horizon Basketball Moves*, ACM TOG / SIGGRAPH Asia 2025. <https://arxiv.org/abs/2509.22442>
- 项目页：<https://pei-xu.github.io/basketball>
- 代码：<https://github.com/xupei0610/basketball>
