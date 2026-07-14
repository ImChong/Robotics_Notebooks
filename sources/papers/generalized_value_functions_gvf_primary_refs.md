# 广义价值函数（GVFs）一手资料索引

> 来源归档（ingest）

- **标题：** Generalized Value Functions (GVFs) 与 Horde 架构经典论文与教程
- **类型：** paper / essay（合集）
- **入库日期：** 2026-07-14
- **一句话说明：** 汇总 GVF 形式化、Horde 并行 off-policy 学习、多时间尺度 nexting 与 GVF 网络等一手来源，作为 `wiki/concepts/generalized-value-functions.md` 的原始依据。
- **沉淀到 wiki：** 是 → [generalized-value-functions](../../wiki/concepts/generalized-value-functions.md)、[richard-sutton](../../wiki/entities/richard-sutton.md)、[model-based-rl](../../wiki/methods/model-based-rl.md)

## 为什么值得保留

- **GVF** 把标准 value function 从「折扣回报」推广到「任意 cumulant 信号的折扣累积」，是 Alberta 学派 **预测性知识表示** 的核心——与当代「世界模型学长期预测」直接对话。
- [Richard Sutton 的 *One-Step Trap*](../blogs/sutton_one_step_trap.md)（2024）明确把 **options + GVFs** 作为一步转移模型 rollout 的替代路线；不收录一手论文则无法追溯该主张的数学与实验来源。
- 机器人侧：Horde 在 **真实移动机器人** 上在线并行学习数千预测；Modayil et al. 的 **nexting** 展示多时间尺度传感器预测——对具身 RL 辅助任务、好奇心与表征学习有直接影响。

## 核心摘录

### 1) Sutton, Precup & Singh (1999) — Options 时序抽象框架

- **来源：** R. S. Sutton, D. Precup, S. Singh, *Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning*, Artificial Intelligence, 112(1–2):181–211, 1999. [作者 PDF](http://incompleteideas.net/papers/sutton-PS98.pdf)
- **要点：**
  - 在 MDP 与 semi-MDP 之间引入 **options**（闭包策略 + 终止条件 + 启动集），把多步行为压缩为「宏动作」。
  - 为后续 GVF 的 **策略条件预测** 与 **时序抽象世界模型** 奠定半马尔可夫结构；Sutton 在 *One-Step Trap* 中与 GVF 并列引用。
- **对 wiki 的映射：** [generalized-value-functions](../../wiki/concepts/generalized-value-functions.md)、[richard-sutton](../../wiki/entities/richard-sutton.md)

### 2) Sutton et al. (2011) — Horde：GVF 并行学习的奠基论文

- **来源：** R. S. Sutton, J. Modayil, M. Delp, T. Degris, P. M. Pilarski, A. White, D. Precup, *Horde: A Scalable Real-time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction*, AAMAS 2011, pp. 761–768. [AAMAS PDF](https://www.ifaamas.org/Proceedings/aamas2011/papers/A6_R70.pdf)
- **要点：**
  - 提出 **Generalized Value Functions (GVFs)**：每个 demon 用独立 **policy π、cumulant（pseudo-reward）、termination γ、terminal-reward z** 定义一个预测问题；语义 grounded 于 sensorimotor 交互。
  - **Horde 架构**：大量 demon **并行 off-policy TD 学习**（GTD 族），每步 **常数时间/内存**，适合机器人实时在线学习。
  - 在配备多传感器的移动机器人上验证：可学 **目标导向行为** 与 **长期预测**，无需手工奖励工程。
- **对 wiki 的映射：** [generalized-value-functions](../../wiki/concepts/generalized-value-functions.md)、[model-based-rl](../../wiki/methods/model-based-rl.md)

### 3) Modayil, White & Sutton (2014) — 多时间尺度 Nexting（GVF 机器人实证）

- **来源：** J. Modayil, A. White, R. S. Sutton, *Multi-timescale Nexting in a Reinforcement Learning Robot*, Adaptive Behavior, 22(5):375–399, 2014. [作者 PDF](http://josephmodayil.com/papers/Modayil-Nexting-AdaptiveBehavior-2014.pdf)
- **要点：**
  - **Nexting**：用 TD(λ) 并行预测 **数千路传感器信号** 在 **0.1–8 s** 等多时间尺度的折扣累积（GVF 特例）。
  - 证明 **span-independent** 学习：预测 horizon 再长，每步更新成本仍与单步 TD 同级——这是 GVF 相对 naive rollout 的关键计算优势。
  - 单一 tile coding 特征即可同时支撑多信号、多尺度预测；30 分钟内达到可观精度。
- **对 wiki 的映射：** [generalized-value-functions](../../wiki/concepts/generalized-value-functions.md)

### 4) van Hasselt & Sutton (2015) — Span Independence 理论

- **来源：** H. van Hasselt, R. S. Sutton, *Learning to Predict Independent of Span*, arXiv:1508.04582, 2015. [arXiv](https://arxiv.org/abs/1508.04582)
- **要点：**
  - 形式化 GVF/TD 学习的 **independence of span** 性质：学习长 horizon 预测时，计算与记忆 **不随 span 线性增长**。
  - 为 Horde / nexting 的「并行学大量长期预测」提供理论支撑；亦被 GVF Networks 等后续工作引用。
- **对 wiki 的映射：** [generalized-value-functions](../../wiki/concepts/generalized-value-functions.md)

### 5) White et al. (2016) — General Value Function Networks (GVFN)

- **来源：** A. White, M. Modayil, R. S. Sutton, *General Value Function Networks*, JAIR 55:265–315, 2016. [JAIR PDF](https://jair.org/index.php/jair/article/download/12105/26653)
- **要点：**
  - 把多个 GVF 的 **question（cumulant + policy + termination）** 与 **answer（函数逼近）** 组织成 **可组合网络**，在 RNN 隐状态中嵌入预测性知识。
  - 论证 GVF 类预测对广泛未来事件的 **表征充分性** 与 **可学习性** 权衡；连接 successor features、预测状态表示等路线。
- **对 wiki 的映射：** [generalized-value-functions](../../wiki/concepts/generalized-value-functions.md)、[foundation-policy](../../wiki/concepts/foundation-policy.md)（辅助任务 / 表征对照）

### 6) Sutton et al. (2023) — Reward-respecting Subtasks（GVF/Options 进入 MBRL）

- **来源：** R. S. Sutton, A. White, M. Modayil, D. Precup, P. M. Pilarski, T. Degris, J. Modayil, M. Delp, *Reward-respecting Subtasks for Model-based Reinforcement Learning*, arXiv:2306.01782, 2023. [arXiv](https://arxiv.org/abs/2306.01782)
- **要点：**
  - 把 **时序抽象子任务** 设计为尊重主任务奖励结构的 options/GVF 式模块，作为 MBRL 中 **一步陷阱** 的结构性替代。
  - Sutton 在 [*One-Step Trap*](../blogs/sutton_one_step_trap.md) 中列为 GVF 路线的最新延伸。
- **对 wiki 的映射：** [generalized-value-functions](../../wiki/concepts/generalized-value-functions.md)、[model-based-rl](../../wiki/methods/model-based-rl.md)

### 7) Sutton (2024) — *The One-Step Trap*（方法论对照博文）

- **来源：** R. S. Sutton, *The One-Step Trap*, Incomplete Ideas, 2024-07-18. [原文](http://incompleteideas.net/IncIdeas/OneStepTrap.html) · 本站归档：[sutton_one_step_trap.md](../blogs/sutton_one_step_trap.md)
- **要点：**
  - 批评 **单步模型 + rollout** 在随机环境下 **误差复合** 与 **指数计算**；主张 **options + GVFs** 构建时序抽象世界模型。
  - 一手列出 Options (1999)、Horde (2011)、Reward-respecting subtasks (2023) 三条引用链。
- **对 wiki 的映射：** [generalized-value-functions](../../wiki/concepts/generalized-value-functions.md)、[model-based-rl](../../wiki/methods/model-based-rl.md)

## 推荐继续阅读（外部）

- Jaderberg et al. (2017) UNREAL — GVF 式辅助任务在深度 RL 中的工程化范例
- Bellemare et al. (2019) GVFs 几何视角 — successor features 与表征几何

## 当前提炼状态

- [x] GVF 七类一手来源摘录与 wiki 映射
- [x] 新建 `wiki/concepts/generalized-value-functions.md`
- [ ] 后续可补：Successor Features、UVFA 与 GVF 的统一视角专节
