# rl_pd_action_interface_locomotion

> 来源归档（ingest）

- **标题：** 腿足 / 人形 RL 中「位置目标 + 底层 PD」动作接口与增益设计 — 代表性论文索引
- **类型：** paper（多篇）
- **入库日期：** 2026-05-12
- **一句话说明：** 围绕「策略输出关节目标、底层 PD/阻抗转力矩」这一工业界长期沿用的接口，归档从四足经典 sim2real、Cassie 双足、全尺寸人形到可变刚度 / 直驱扭矩等路线的公开论文与报告链接，供 [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md) 与各篇 **wiki 实体子页**（`wiki/entities/paper-*.md`）交叉引用。
- **沉淀到 wiki：** 是 → [Kp/Kd query 页](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md) + 10 个 `wiki/entities/paper-*.md` 子页（各含 Mermaid）

## 为什么值得保留

同一套 RL 算法在不同「动作语义」（目标角、残差、扭矩、刚度参数）下会学到完全不同的接触与带宽行为；本索引把 **PD 内环在文献中的公开数值、控制分频与随机化范围** 与 **何时考虑弃用 PD** 的代表作放在一起，避免只从单一代码库抄默认 `stiffness/damping` 而缺乏文献对照。

## 核心论文摘录

### 1) Real-World Humanoid Locomotion with Reinforcement Learning（Digit，全尺寸人形）

- **链接：** [arXiv:2303.03381](https://arxiv.org/abs/2303.03381) · [项目页](https://learning-humanoid-locomotion.github.io/)
- **核心贡献：** 大规模并行仿真 + 域随机化训练因果 Transformer 策略，从本体感觉与动作历史自回归预测下一步动作，在 Agility Digit 上零样本户外行走；强调 sim2real 流水线（Isaac Gym 训练 → 厂商高保真仿真校验 → 真机）。
- **与 Kp/Kd：** 动作链仍为「学习器输出关节级指令 → 低层跟踪（含关节阻抗）」范式；论文附录 / 补充材料给出仿真与部署中的 **关节 PD 增益表**（随关节而异）。公开技术讨论中常以 **髋部量级 \(K_p \approx 200\) N·m/rad、\(K_d \approx 10\) N·m·s/rad** 作为可读锚点 — **精读时请以 PDF 表格与随文代码为准逐关节核对**。
- **对 wiki 的映射：**
  - [论文实体：Digit 人形 RL 行走](../../wiki/entities/paper-digit-humanoid-locomotion-rl.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)

### 2) Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control（Cassie）

- **链接：** [arXiv:2401.16889](https://arxiv.org/abs/2401.16889)
- **核心贡献：** 双历史（长/短 horizon）I/O 架构 + 任务随机化，在 Cassie 上统一多种动态技能并直接 sim2real。
- **与 Kp/Kd：** 文中给出 **策略控制频率约 33 Hz**、**关节 PD 内环约 2 kHz** 的分工；训练中对默认 PD 增益做 **缩放随机化（文中为标称的约 0.7–1.3 倍）**，用于回答「固定标称增益是否足够」与「随机化范围取多大」的工程问题。
- **对 wiki 的映射：**
  - [论文实体：Cassie 双足多技能 RL](../../wiki/entities/paper-cassie-biped-versatile-locomotion-rl.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)
  - [Domain Randomization 指南](../../wiki/queries/domain-randomization-guide.md)

### 3) Variable Stiffness for Robust Locomotion through Reinforcement Learning

- **链接：** [arXiv:2502.09436](https://arxiv.org/abs/2502.09436)
- **核心贡献：** 将 **可变刚度** 与关节位置一起纳入策略动作；比较 **逐关节 / 分腿 / 混合** 等参数化方式；户外平地仅训练即可迁移到多种户外地形。
- **与 Kp/Kd：** 直接回应「RL 能否学增益/刚度」：**能学刚度，但分组建模往往更稳**；阻尼侧常保留 **结构约束或物理一致关系**，避免与动力学任意解耦。
- **对 wiki 的映射：**
  - [论文实体：可变刚度腿足 RL](../../wiki/entities/paper-variable-stiffness-locomotion-rl.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)

### 4) Learning Locomotion Skills for Cassie: Iterative Design and Sim-to-Real

- **链接：** [arXiv:1903.09537](https://arxiv.org/abs/1903.09537) · [CoRL 2019 PMLR](http://proceedings.mlr.press/v100/xie20a.html)
- **核心贡献：** 记录 **奖励、观测、动作接口多轮迭代** 的 Cassie 行走 RL 设计过程；引入 DASS 等机制在奖励重写时复用旧策略经验。
- **与 Kp/Kd：** 强调 **接口与 reward 共同演化** — PD 增益若与观测归一化、动作缩放不同步改，会反复触发「仿真能走、真机不能」的调试循环。
- **对 wiki 的映射：**
  - [论文实体：Cassie 迭代式 sim2real](../../wiki/entities/paper-cassie-iterative-locomotion-sim2real.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 5) Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning（ANYmal + legged_gym）

- **链接：** [arXiv:2109.11978](https://arxiv.org/abs/2109.11978) · [legged_gym 项目页](https://leggedrobotics.github.io/legged_gym/)
- **核心贡献：** 单机 GPU 上千并行环境 + 课程式地形，在数分钟内训练 ANYmal 平地策略、约二十分钟粗糙地形；开源 **legged_gym** 成为后续 RL+PD 配置的「教科书式」基线。
- **与 Kp/Kd：** 与 [`legged_gym` 配置中的 `stiffness`/`damping` / `decimation`](../../wiki/entities/legged-gym.md) 直接对应，适合做 **增益扫描与消融** 的公共起点。
- **对 wiki 的映射：**
  - [论文实体：ANYmal 分钟级并行 DRL](../../wiki/entities/paper-anymal-walk-minutes-parallel-drl.md)
  - [legged_gym](../../wiki/entities/legged-gym.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)

### 6) Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior

- **链接：** [arXiv:2212.03238](https://arxiv.org/abs/2212.03238) · [项目页](https://gmargo11.github.io/walk-these-ways/)
- **核心贡献：** 单一策略内嵌 **多种步态/风格（MoB）**，通过少量行为参数在部署时快速切换，减轻「每换一种地形就重训 reward」的迭代成本；开源四足控制器。
- **与 Kp/Kd：** 在已能跑通 legged_gym 式固定 PD 后，展示如何把 **增益随机化、低层安全与部署期人类调参** 接到同一套 sim2real 叙事里，以提升 **分布外泛化**。
- **对 wiki 的映射：**
  - [论文实体：Walk These Ways（MoB）](../../wiki/entities/paper-walk-these-ways-quadruped-mob.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)

> **易混提示：** `2212.03238` 的标题是 **Walk These Ways**，不是「Learning to Walk in Minutes」；后者见上一条 **2109.11978**。

### 7) Feedback Control For Cassie With Deep Reinforcement Learning

- **链接：** [arXiv:1803.05580](https://arxiv.org/abs/1803.05580)
- **核心贡献：** 在 **高保真 Cassie 模型** 上用 DRL 学习反馈跟踪参考运动；讨论延迟、盲走不规则地形与扰动下的鲁棒性。
- **与 Kp/Kd：** 为「**为何常用 PD 目标空间而非直接扭矩**」提供早期、清晰的 **MDP 表述 + 仿真—硬件对齐** 叙述：低维目标 + 已知内环比端到端扭矩更易稳定优化。
- **对 wiki 的映射：**
  - [论文实体：Cassie 反馈控制 DRL](../../wiki/entities/paper-cassie-feedback-control-drl.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)

### 8) Learning Torque Control for Quadrupedal Locomotion

- **链接：** [arXiv:2203.05194](https://arxiv.org/abs/2203.05194)
- **核心贡献：** 策略 **直接输出关节扭矩**（高频），弱化或绕过固定 PD 内环，在 sim2real 上展示动态与抗扰潜力。
- **与 Kp/Kd：** 用于判断 **「我是否应放弃 PD 先验」** — 若你认为固定 PD 是多余结构，应先用此文与后续扭矩控制文献对照 **带宽、安全滤波与训练难度** 的代价。
- **对 wiki 的映射：**
  - [论文实体：四足扭矩控制 RL](../../wiki/entities/paper-quadruped-torque-control-rl.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 9) Sim-to-Real: Learning Agile Locomotion For Quadruped Robots（四足敏捷 locomotion 经典）

- **链接：** [RSS 2018 Proceedings PDF（p10）](https://www.roboticsproceedings.org/rss14/p10.pdf)
- **核心贡献：** 随机化动力学与传感，在仿真中学 **高频扭矩/力矩式** 敏捷运动并迁移到实物四足，建立后续「位置目标 + 低层 PD」与「直接力矩」两条线争论的 **历史参照系**。
- **与 Kp/Kd：** 帮助建立 **「为何位置/扭矩接口在工业与论文中长期并存」** 的直觉：不同硬件带宽与安全需求会锁定不同接口层。
- **对 wiki 的映射：**
  - [论文实体：RSS 2018 敏捷四足 sim2real](../../wiki/entities/paper-quadruped-agile-sim2real-rss2018.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)

### 10) Learning Variable Impedance Control for Contact Sensitive Tasks

- **链接：** [arXiv:1907.07500](https://arxiv.org/abs/1907.07500) · IEEE RA-L 2020
- **核心贡献：** 在接触敏感任务中，让策略同时学 **期望轨迹与关节空间阻抗参数**，并用正则改善可解释性与迁移；实验含弹跳与桌面擦拭。
- **与 Kp/Kd：** 为后续 **可变刚度 locomotion** 提供「**位置 + 阻抗参数联合输出**」的思想前史；强调 **阻抗 shaping** 对接触 RL 样本效率的影响。
- **对 wiki 的映射：**
  - [论文实体：可变阻抗接触任务 RL](../../wiki/entities/paper-variable-impedance-contact-rl.md)
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)
  - [Force Control Basics](../../wiki/concepts/force-control-basics.md)

## 当前提炼状态

- [x] 已建立 10 篇论文的一手链接与与 Kp/Kd / 动作接口相关的映射句；每篇均有独立 **wiki 实体子页**（`wiki/entities/paper-*.md`）
- [~] 后续可按「仅人形 / 含操作臂接触」拆分子索引，并补厂商阻抗表链接
