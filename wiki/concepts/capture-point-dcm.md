---
type: concept
status: complete
summary: "Capture Point / DCM 用线性倒立摆的发散分量描述步行平衡，是步位规划和扰动恢复的核心直觉工具。"
updated: 2026-11-07
original:
  - ../../sources/papers/lip_zmp_dcm_walking_reading_primary_papers.md
related:
  - ../entities/paper-faststair-humanoid-stair-ascent.md
  - ../entities/paper-fddc.md
  - ../entities/paper-fail-passive-gap.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ./footstep-planning.md
sources:
  - ../../sources/papers/lip_zmp_dcm_walking_reading_primary_papers.md
  - ../../sources/papers/whole_body_control.md
  - ../../sources/papers/footstep_and_balance.md
  - ../../sources/papers/faststair_arxiv_2601_10365.md
  - ../../sources/papers/fddc_arxiv_2608_00500.md
  - ../../sources/papers/fail_passive_gap_arxiv_2608_02809.md
---

# Capture Point / DCM

**Capture Point（捕获点）** 和 **DCM（Divergent Component of Motion，发散运动分量）** 是腿式机器人动态平衡与步态控制里两个非常关键的概念，用来描述系统“往哪里倒”以及“该踩到哪里才能救回来”。

## 一句话定义

- **Capture Point**：如果机器人已经在往前倒，那么要想在有限步内停住，脚应该踩到哪里。
- **DCM**：把系统运动拆成“会发散的那一部分”，方便分析动态平衡和控制。

一句话说白了：

> ZMP 更像“现在还能不能稳住”，Capture Point / DCM 更像“已经开始失稳时，下一步该怎么救”。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DCM | Divergent Component of Motion | 质心发散分量，用于落脚点与平衡调节 |
| ZMP | Zero Moment Point | 足式平衡判据，地面反力合力矩为零的点 |
| LIP | Linear Inverted Pendulum | 线性倒立摆，质心动力学的常用简化模型 |
| CoM | Center of Mass | 质心，平衡与 locomotion 规划的核心状态量 |
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习训练框架 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 为什么重要

LIP / ZMP 是理解双足步行的经典入口，但它们有一个局限：

- ZMP 约束回答的是“当前接触力矩是否可行”，本身不告诉你质心的发散趋势
- 对强扰动恢复、需要靠迈步才能稳住的情形不够直观（带 ZMP 约束的 MPC 并不比不带约束的 DCM 反馈控制更能提高速度，见下文 iCub 对比）

一旦机器人已经在往前倒，光说“ZMP 要在支撑多边形里”就不够了。

这时候更核心的问题变成：

- 系统当前的发散趋势往哪里走
- 是否还能通过下一步踩踏把自己接住
- 如果能接住，脚应该放到哪里

这正是 Capture Point / DCM 的核心价值。

## 核心原理：背后的模型直觉

这些概念通常建立在 **LIP（线性倒立摆）** 模型上。

LIP 的经典动力学：

$$
\ddot{x} = \omega^2 (x - x_{zmp}), \quad \omega = \sqrt{g / z_c}
$$

这里：
- \( x \)：质心位置
- \( \dot{x} \)：质心速度
- \( x_{zmp} \)：ZMP
- \( z_c \)：质心高度
- \( \omega \)：倒立摆自然频率

这个系统很重要的一点是：
- 它包含**稳定模式**和**发散模式**
- Capture Point / DCM 正是在抓“发散那一支”

**核对原文后的精确表述（Romualdi 等 2018；Khadiv 等 2016）：** 令 $\xi = x + \dot{x}/\omega$，LIP 可拆成

$$
\dot{x} = \omega(\xi - x),\qquad \dot{\xi} = \omega(\xi - x_{zmp})
$$

第一式说明 CoM 以时间常数 $1/\omega$ 追随 DCM（稳定），第二式说明 DCM 以正实根 $\omega$ 发散，只能靠 $x_{zmp}$（或迈步改变支撑点）去压。常值 ZMP 下的解析解为 $\xi(t)=x_{zmp}+e^{\omega t}(\xi_0-x_{zmp})$；ZMP 与 DCM 的关系为 $x_{zmp}=\xi-\dot{\xi}/\omega$。三维情形下 $\xi = x + b\dot{x}$，$b=\sqrt{z_0/g}$，用虚拟排斥点 VRP 写成 $\dot{\xi}=(\xi-r_{vrp})/b$；质心角动量恒定时 VRP 在地面的投影即 ZMP（Shafiee 等 2019）。Romualdi 等把 DCM 看作 Capture Point 向三维的推广（前提是 VRP 与 eCMP 高度差恒定）。

## Capture Point 是什么

最经典的一维定义是：

$$
\xi = x + \frac{\dot{x}}{\omega}
$$

其中：
- \( x \)：质心位置
- \( \dot{x} \)：质心速度
- \( \omega = \sqrt{g/z_c} \)

这个量 \( \xi \) 就是 Capture Point。

它的物理意义可以理解为：

> 如果你此刻想把机器人停住，并且允许它再迈一步，那么脚应该踩向的位置大致就是这个点。

### 直觉理解

- 如果机器人站得很稳，速度很小，Capture Point 靠近 CoM
- 如果机器人往前冲得很快，Capture Point 会跑到前面很远
- 如果 Capture Point 已经跑出支撑区域，而你又不能迈步，那大概率稳不住

## DCM 是什么

DCM（Divergent Component of Motion）在很多文献里和 Capture Point 非常接近，甚至在 LIP 条件下它们本质上就是一个东西。（历史归属据 Khadiv 等 2016 引言，属二手：Hof 以 extrapolated CoM 之名、Pratt 以 Capture Point 之名早已用过同一量，Takenaka 等用它生成轨迹，Englsberger 等提出对 DCM 不稳定部分做控制；本库此前把 DCM 归给 Koolen 等的说法未被核实。）

它定义为：

$$
\xi = x + \frac{\dot{x}}{\omega}
$$

这个变量有个关键动力学：

$$
\dot{\xi} = \omega (\xi - x_{zmp})
$$

这个式子非常有启发性。

它说明：
- DCM 自己会发散
- ZMP 可以用来“拉住”这个发散趋势
- 步态控制很多时候其实就是在控制 DCM

## 为什么 DCM 比单看 CoM 更有用

如果你只看 CoM：
- 会知道机器人现在在哪里
- 但不一定知道它“接下来会往哪里跑”

如果你看 DCM：
- 你会同时考虑位置和速度
- 更接近系统真正的失稳趋势

所以 DCM 的好处是：

> 它把“状态 + 趋势”压进了一个更适合动态平衡控制的变量里。

## Capture Point / DCM 和 ZMP 的关系

这三个概念很容易混。

### ZMP
更像：
- 当前支撑稳定性相关量
- 地面合力矩平衡点
- 常用来做经典步态规划和稳定性约束

### Capture Point / DCM
更像：
- 当前运动发散趋势
- 如果失稳，下一步该踩哪里
- 更适合做动态平衡和扰动恢复分析

一句话：

> ZMP 更像“当前还能不能稳住”，Capture Point / DCM 更像“已经开始不稳时该怎么接住”。

见：[LIP / ZMP](./lip-zmp.md)

## 和 Contact Dynamics 的关系

Capture Point / DCM 看起来像质心变量，但本质上依赖接触能力。

因为你能不能把自己接住，取决于：
- 能不能在合适时刻建立新接触
- 能不能在可行位置落脚
- 接触力和摩擦约束是否允许

所以它不是脱离 contact 单独存在的量。

见：[Contact Dynamics](./contact-dynamics.md)

## 和 Floating Base Dynamics 的关系

在 floating base 系统里，base 不可直接控制，整体稳定性非常依赖：
- 接触组织
- 质心速度
- 动量演化

Capture Point / DCM 是对这类发散趋势的简化描述。

见：[Floating Base Dynamics](./floating-base-dynamics.md)

## 工程实践：在机器人中的典型应用

### 0. 一个已发表的 DCM 三层架构（Romualdi 等 2018；Shafiee 等 2019）
- **轨迹优化层**：把机器人近似为独轮车规划落脚点与时刻，单支撑内按 $\xi(t)=r_{zmp}+e^{\omega(t-t_i)}(\xi^{eos}-r_{zmp})$ 反向递推各步终点 DCM；双支撑内用三次多项式保证 $C^1$ 连续，进而 ZMP 连续。
- **简化模型控制层**：瞬时律 $r^{ref}=\xi_{ref}-\dot{\xi}_{ref}/\omega+K_p(\xi-\xi_{ref})+K_i\!\int(\xi-\xi_{ref})$，$K_p>I$ 时误差渐近收敛；缺点是不保证 ZMP 留在支撑多边形内、不看未来。预测律把 ZMP 作输入、加多边形线性约束，解 QP。
- **全身 QP 层**：把期望 VRP 换成质心加速度 / 线动量变化率，在位置 / 速度 / 力矩模式下跟踪。
- **iCub 实测对比**（最高直行速度）：预测+速度控制 0.19 m/s、预测+位置控制 0.20 m/s、瞬时+速度控制 0.22 m/s、瞬时+位置控制 **0.41 m/s**；作者发现预测律的期望 ZMP 变化过快，对 ZMP 跟踪误差更不鲁棒。
- **推扰恢复**（Shafiee 等 2019，仿真）：iCub（33 kg）0.28 m/s 行走，盆骨受 150 N、0.05 s 侧推，step adapter 在单支撑里用一个 QP 同时调整步位与步时（平均步宽增 6 cm，步时较名义 0.53 s 缩短约 0.12 s），整套架构平均 < 3 ms。
- **步时也要调**（Khadiv 等 2016，LIPM 仿真）：60 kg、质心 0.8 m，限制步长 ±50 cm、步时 0.2–0.8 s，受 325 N、0.1 s 推力时固定步时无法恢复，联合步位 + 步时可恢复；点足 / 被动踝机器人无法靠 CoP 调节 DCM，因此迈步更关键。

### 1. 扰动恢复
机器人被推一把之后：
- 是否需要迈步
- 往哪迈
- 多远能接住

这些问题几乎都可以借助 Capture Point 理解。

### 2. 动态步态控制
比起只用 ZMP，很多更动态的 walking controller 会显式或隐式使用 DCM / CP。

### 3. Footstep Planning
Capture Point 可以直接影响：
- 下一步落脚位置
- 步长调整
- 何时从站立转入迈步

### 4. 高动态 locomotion 直觉建立
如果你想理解“为什么机器人不能一直原地稳住，而必须迈步”，Capture Point 是最好的概念之一。

## 在人形控制里的意义

对于人形机器人来说：
- 站立时可以主要看静态 / 准动态平衡
- 一旦开始快速行走、跑、被推、切换支撑，动态因素立刻主导

这时：
- Capture Point 告诉你“还能不能接回来”
- DCM 告诉你“发散分量现在往哪里走”

所以在现代 humanoid locomotion 语境里，它们往往比单独讲 ZMP 更贴近控制直觉。

### 5. GPU 并行 DCM 落脚点规划（FastStair）

[FastStair（arXiv:2601.10365）](../entities/paper-faststair-humanoid-stair-ascent.md) 把 **变高倒立摆（VHIP）** 上的 DCM 演化改写为 **离散候选集上的并行代价评估 + argmin**，用张量化 GPU 运算嵌入 Isaac Lab 大规模 RL——在保留 **接触可行性与平衡结构** 的同时，避免纯 model-free RL 在楼梯上的不安全行为。这与 [Footstep Planning](./footstep-planning.md) 中「DCM 触发步位更新」的经典路线不同：FastStair 让 DCM 规划器 **直接成为训练期特权监督**，而非仅作在线 MPC 模块。

## 局限与常见误区

### 0. 以为 DCM 控制不需要模型假设
上面的结论都建立在 LIP / VRP 简化上；Shafiee 等自述其步位调整“只用 DCM 简化模型，不考虑角动量，大扰动可能需要对名义轨迹做过大修改”，且只在仿真里验证。DCM 瞬时律还默认 ZMP 可以随意设置，而点足 / 被动踝机器人根本无法调 CoP。

### 1. 以为 Capture Point 能完全替代 ZMP
不能。它们关注的物理问题不同，常常是互补关系。

### 2. 以为 Capture Point 是精确万能落脚解
不是。它是很有力的简化指标，但真实系统还要考虑接触、关节限位、可达性、时序等。

### 3. 以为 DCM 是一个完全不同于 Capture Point 的新东西
在很多 LIP 语境里，它们本质上非常接近。区别更多在使用语境和控制表达方式。

### 4. 低估速度项的重要性
很多初学者只盯着 CoM 位置，但动态平衡真正关键的是“位置 + 速度”的组合。

### 5. 以为 xCoM 只能进奖励 / 特权 critic
世界系 capture point 确实需要基座线速度 \(v_b\)，机载常测不到。但相对**支撑足**改写后 \(v_b\) 相消，可只靠编码器+IMU 重构并直接进部署 actor——见 [FDDC](../entities/paper-fddc.md)（单腿基准上消融掉动态 CoM 观测独掉 40 pt Perfect）。

### 6. 保护停可以「立刻冻住」
工业急停若打在单支撑，立即冻结通常不可行：必须先把当前步落到可捕获点再静站。这给机械停时 \(t_{\mathrm{stop}}\) 一个相位相关下界，也是人形功能安全间距必须按最坏相位定的原因，见 [Fail-Passive Gap](../entities/paper-fail-passive-gap.md)。

## 推荐使用建议

### 如果你学 locomotion
一定值得掌握。因为它能把“为什么要迈步”这件事讲得非常直观。

### 如果你学 MPC / footstep planning
Capture Point / DCM 是很重要的中间概念。

### 如果你学 WBC / TSID
哪怕你不直接用它们做控制变量，也建议理解，因为它们能帮你建立更动态的稳定性直觉。

## 参考来源

- **原文通读归档：** [LIP / ZMP / DCM 四篇一手论文通读](../../sources/papers/lip_zmp_dcm_walking_reading_primary_papers.md) — Romualdi 2018、Shafiee 2019、Khadiv 2016、Herdt 2010，全文已读
- **以下三篇经典原文本次未取得（付费墙），内容只经上述论文引述，属二手：**
- Pratt et al., *Capture Point: A Step toward Humanoid Push Recovery* (2006) — Capture Point 概念提出
- Englsberger et al., *Three-Dimensional Bipedal Walking Control Based on Divergent Component of Motion* (2015) — DCM 三维扩展
- Koolen et al., *Capturability-based Analysis and Control of Legged Locomotion* (2012) — 可捕获性理论
- **ingest 档案：** [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md)
- **ingest 档案：** [sources/papers/footstep_and_balance.md](../../sources/papers/footstep_and_balance.md) — Kajita ZMP / Pratt CP / Koolen DCM 核心论文
- **ingest 档案：** [FDDC（arXiv:2608.00500）](../../sources/papers/fddc_arxiv_2608_00500.md) — 支撑相对动态 CoM 进可部署 actor + 单腿 sim2sim 基准
- **ingest 档案：** [Fail-Passive Gap（arXiv:2608.02809）](../../sources/papers/fail_passive_gap_arxiv_2608_02809.md) — 保护停的单支撑下界与 ISO 13855 最坏 \(t_{\mathrm{stop}}\)

## 关联页面

- [LIP / ZMP](./lip-zmp.md) · [ZMP + LIP 形式化](../formalizations/zmp-lip.md) · [Footstep Planning](./footstep-planning.md) · [Balance Recovery](../tasks/balance-recovery.md) · [Contact Dynamics](./contact-dynamics.md)

## 推荐继续阅读

- Romualdi et al., *A Benchmarking of DCM Based Architectures…* <https://arxiv.org/abs/1809.02167>；Shafiee et al., *Online DCM Trajectory Generation for Push Recovery…* <https://arxiv.org/abs/1909.10403>；Khadiv et al., *Step Timing Adjustment* <https://arxiv.org/abs/1610.02377>
- Pratt et al., *Capture Point: A Step toward Humanoid Push Recovery*
- Englsberger et al., *Three-Dimensional Bipedal Walking Control Based on Divergent Component of Motion*
- [LIP / ZMP](./lip-zmp.md)
- [Contact Dynamics](./contact-dynamics.md)
- [Footstep Planning](./footstep-planning.md) — DCM 是步位规划稳定性的核心依据
- [Balance Recovery](../tasks/balance-recovery.md)（Capture Point 的实际应用场景：扰动恢复策略）
- [FDDC](../entities/paper-fddc.md) — 可部署动态 CoM 观测与单腿平衡基准（arXiv:2608.00500）
- [Fail-Passive Gap](../entities/paper-fail-passive-gap.md) — 单支撑保护停必须先 capture，再谈认证间距

## 一句话记忆

> Capture Point / DCM 研究的是“机器人当前的发散趋势往哪里走，以及下一步该踩到哪里才能把自己接住”，它是比 ZMP 更动态的平衡与扰动恢复理解框架。
