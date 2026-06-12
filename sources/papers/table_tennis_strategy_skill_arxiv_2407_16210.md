# Strategy and Skill Learning for Physics-based Table Tennis Animation

> 来源归档（ingest）

- **标题：** Strategy and Skill Learning for Physics-based Table Tennis Animation
- **项目名：** **PhysicsPingPong**
- **类型：** paper / physics-based character animation / multi-agent interaction / VR
- **arXiv：** <https://arxiv.org/abs/2407.16210>（PDF：<https://arxiv.org/pdf/2407.16210>）
- **会议：** SIGGRAPH 2024 Conference Papers（DOI：[10.1145/3641519.3657437](https://doi.org/10.1145/3641519.3657437)）
- **项目页：** <https://jiashunwang.github.io/PhysicsPingPong/>
- **代码：** <https://github.com/jiashunwang/PhysicsPingPong>（处理后的 motion 与数据链接见 README；完整代码需联系作者）
- **作者：** Jiashun Wang, Jessica Hodgins, Jungdam Won
- **机构：** Carnegie Mellon University, The AI Institute, Seoul National University
- **入库日期：** 2026-06-12
- **一句话说明：** **分层控制**：技能层用 **5 路 ASE 模仿 + ball-control + mixer 混合** 缓解 mode collapse；策略层用 **迭代行为克隆（CVAE）** 在 agent–agent 与 **VR 人–机** 环境中学习技能选择与落点，支持竞争/合作两种策略。

## 核心摘录

### 1) 技能层（Skill-level）

三阶段训练：

1. **Imitation policies** \(\pi^i\)：5 种乒乓球技能子集 + 1 个 universal 策略；基于 **ASE** 对抗框架 + 超球面 latent \(z^i\)。
2. **Ball control policies** \(\omega^i\)：在随机发球下用对应模仿策略把球打到目标落点；奖励 = 拍面接近 \(r_p\) + 落点 \(r_b\) + style \(r_s\)。
3. **Mixer policy** \(\omega^m\)：对 universal 与 5 技能动作做 **关节级混合权重** \(\varphi\)，实现技能间快速过渡（式 8）。

**五种技能**：正手攻球、正手搓球、正手扣杀、反手攻球、反手搓球。

### 2) 策略层（Strategy-level）

- 输入：己方状态 \(s\)、对手 \(\tilde{s}\)、球 \(b\)。
- 输出：one-hot 技能指令 \(\delta\) + 目标落点 \(y\)（每次对手来球更新一次）。
- **迭代 BC**：随机或视频启发式策略采集对局数据 → CVAE 拟合 → 用新策略再采集（Algorithm 1）；竞争策略筛「获胜」片段，合作策略筛「对手成功回球」片段。

### 3) 交互环境

| 环境 | 说明 |
|------|------|
| **Agent–agent** | 双虚拟智能体对打；对手用随机或 **broadcast video** 训练的启发式 CVAE |
| **Human–agent VR** | Unity 渲染 + Isaac Gym 仿真；VR 手柄位姿驱动用户球拍刚体，实现实时全身动力学对打 |

### 4) 实验要点

- 相对 **ASE / CASE / ET（无 mixer）**：Discriminator Score、Skill Accuracy、Diversity Score 更高；平均连续回球 **10.93**（ASE 9.54）。
- Mixer 在触球瞬间 \(\varphi\) 最低（依赖预训练 ball-control），过渡阶段 \(\varphi\) 升高。

## 对 wiki 的映射

- 新建方法页：[`wiki/methods/table-tennis-strategy-skill-learning.md`](../../wiki/methods/table-tennis-strategy-skill-learning.md)
- 交叉更新：
  - [`wiki/entities/smplolympics.md`](../../wiki/entities/smplolympics.md) — 同生态乒乓球 baseline 环境
  - [`wiki/methods/imitation-learning.md`](../../wiki/methods/imitation-learning.md) — 技能层模仿学习背景
  - [`wiki/methods/ase.md`](../../wiki/methods/ase.md) — 技能模仿基线
  - [`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md) — VR 人–机对打接口

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2407.16210>
- 项目页：<https://jiashunwang.github.io/PhysicsPingPong/>
- 代码：<https://github.com/jiashunwang/PhysicsPingPong>
