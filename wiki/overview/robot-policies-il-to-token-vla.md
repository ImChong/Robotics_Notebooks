---
type: overview
tags: [overview, robot-learning, imitation-learning, reinforcement-learning, vla, action-chunking, control, flow-matching, world-models]
status: complete
updated: 2026-11-25
original: ../../sources/courses/asu_cse598_fall2026_physical_ai_robot_policies.md
summary: "对照 ASU CSE 598（Fall 2026）71 页官方课件全文核读编译：以统一加权对数似然框架串联控制契约、模仿学习、深度强化学习、Token VLA、Diffusion/Flow 策略与 World Models，并给出四层控制频率栈与真机排查方法。"
related:
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../methods/vla.md
  - ../methods/action-chunking.md
  - ../methods/diffusion-policy.md
  - ../methods/generative-world-models.md
  - ../concepts/sim2real.md
  - ../comparisons/robot-learning-five-paradigms-taxonomy.md
  - ../../roadmap/depth-vla.md
  - ../../roadmap/depth-imitation-learning.md
sources:
  - ../../sources/courses/asu_cse598_fall2026_physical_ai_robot_policies.md
  - ../../sources/blogs/wechat_human_five_robot_policies_2026-11-24.md
---

# Robot Policies: From Imitation Learning to Token VLA / 机器人策略与 Physical AI 全栈总览

> 本页基于 Xinyuan Liu 在 Arizona State University CSE 598（Fall 2026）的 Physical AI 客座讲座 [官方 71 页课件](../../sources/courses/asu_cse598_fall2026_physical_ai_robot_policies.md) 全文核读编译，并结合 [human five 课程笔记](../../sources/blogs/wechat_human_five_robot_policies_2026-11-24.md) 与已归档单篇论文原文交叉整理。课程以“pick up the red cup”（UMI RSS 2024 示例）贯穿 7 个板块，补充了二手中文笔记缺失的 Diffusion/Flow、World Models 与真机系统集成部分，并更正了个别模型出处。

## 一句话定义

机器人策略将观测和目标映射到可执行动作；IL 与 RL 主要区别在训练信号，VLA 是条件输入与策略架构的一种实现方式，三者并非互斥。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| IL | Imitation Learning | 模仿专家示范 |
| BC | Behavior Cloning | 监督观测到动作 |
| RL | Reinforcement Learning | 优化长期回报 |
| ACT | Action Chunking with Transformers | 预测动作序列 |
| PPO | Proximal Policy Optimization | 近端策略优化 |
| VLA | Vision-Language-Action | 视觉语言条件动作策略 |
| FK / IK | Forward / Inverse Kinematics | 正 / 逆运动学 |
| DCT | Discrete Cosine Transform | 时序动作频域变换 |
| BPE | Byte Pair Encoding | 压缩离散符号序列 |

## 为什么重要

将任务语义、训练监督、动作分布和实时控制分层，才能定位错误究竟来自数据覆盖、模型输出、动作坐标系、推理延迟，还是控制器/执行器。学习路线不等同于模型版本演进。

## 核心原理与流程总览

```mermaid
flowchart TD
  G["任务目标或语言指令"] --> P["策略：观测到动作"]
  O["相机、关节和其他观测"] --> P
  P --> A["单步动作或动作块"]
  A --> S["动作解码、参考系转换、调度"]
  S --> C["底层控制器与驱动器"]
  C --> R["机器人与环境"]
  R --> O
  D["专家示范与纠错"] --> I["IL：监督更新"]
  R --> E["Rollout 与奖励"]
  E --> L["RL：回报优化"]
  I --> P
  L --> P
  V["视觉语言预训练"] --> P
```

训练分支并不意味着部署时每一步都更新策略参数；冻结策略也可闭环执行。

### 1. Physical AI 与学习问题

状态描述系统真实情况，观测是传感器可获得的信息；部分可观测时可加入历史、记忆或状态估计。轨迹是一系列状态和动作，rollout 是运行产生轨迹的过程，episode 通常按任务终止或时限定义，三词并非严格同义。

以下为标准形式的补充，不是公众号中丢失的公式原样恢复：

- 策略：$a_t\sim\pi_\theta(\cdot\mid o_{0:t},g)$。
- 折扣回报：$G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}$；期望回报 $J(\theta)=\mathbb E[G_0]$。
- 完全可观测的固定目标 MDP：$p_\theta(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$。
- **统一训练视角（Slide 7）：** $\max_\theta \mathbb{E}_{(o,a)\sim q}[w(o,a)\log \pi_\theta(a\mid o)]$，用数据分布 $q$ 与样本权重 $w$ 统一对比多种范式：BC/ACT/DP/VLA（专家示范，$w=1$）、DAgger（策略访问状态 + 专家标注，$w=1$）、Filtered BC（自身 rollout，成功为 $1$ 否则 $0$）、PPO（自身 rollout，$w$ 为优势函数 $A$）及 $\pi^*_{0.6}$ Recap（示范 + 自身数据，以优势信号作为条件）。

### 2. 动作表示与控制契约

| 维度 | 选项 | 部署核对 |
|---|---|---|
| 空间 | Joint / Task | 关节顺序、末端参考系、IK 多解与可达性 |
| 参考 | Absolute / chunk-start Relative / stepwise Delta | 参考时刻、坐标系、积分与姿态组合；各框架命名可能不同 |
| 模式 | Position / Velocity / Impedance / Torque | 单位、限位、控制器增益、饱和、接触稳定性 |
| 时间 | 单步 / chunk | 采样频率、执行前缀、观测时间戳 |

策略输出不等同于电机电流。对于典型关节 PD 目标控制，可用 $\tau=K_p(q_d-q)+K_d(\dot q_d-\dot q)+\tau_{ff}$ 说明目标到力矩的接口；真实驱动器结构与控制模式依硬件而定。阻抗控制强调期望动态关系，不只意味着降低位置增益。

Sim2Real 可从域随机化、真机数据混训、生成式数据扩增和测量域差切入；生成场景外观不等同于物理动态已校准。

### 3. 模仿学习

| 方法 | 解决的问题 | 仍有的约束 |
|---|---|---|
| BC + MSE | 监督回归 | 多模态平均、分布偏移 |
| GMM | 表达多个动作模式 | 模式覆盖、采样质量 |
| BC-RNN | 利用历史推断运动趋势 | 隐状态、历史长度 |
| [ACT](../entities/paper-act.md) | CVAE 与 Transformer 预测连续动作块 | 部署反馈频率、跨块一致性 |
| DAgger | 收集 learner 访问状态的专家标签 | 专家查询成本与安全 |
| HG-DAgger | 人类决定接管及交还控制 | 接管判据、纠错数据覆盖 |

ACT 的训练后验编码动作序列与状态；其经典部署设置使用零潜变量。时间集成融合对同一目标时刻的多次预测，不能直接平均不同目标时刻的动作。

### 4. 强化学习

Rollout 收集状态、动作和奖励；actor 生成动作，critic 估计价值，优势函数 $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ 比较动作相对基线的收益。PPO-Clip（课件以 Unitree Go2 `MimicKit` 与 $\epsilon=0.2$ 为例）优化裁剪代理目标 $L(\theta)=\mathbb{E}_t[\min(r_t A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]$；裁剪限制目标中的更新激励，并不是对新旧策略 KL 或所有概率比的硬约束。

AMP（SIGGRAPH 2021 + MimicKit G1 侧手翻示例）使用判别器学习运动风格奖励，并与任务奖励组合。LLM/VLM 自动奖励方面，课件对比了 Eureka（GPT-4 生成可微/可运行奖励代码）与 **ROOT**（LENS Lab 工作，以 VLM 审视 rollout 视频优化奖励函数：$\max_\psi F(\pi^*_\psi)$ s.t. $\pi^*_\psi=\arg\max_\pi J(\pi; r_\psi)$）；工程上仍需核对真实任务成功、奖励投机、接触安全与重复试验，不能只优化 VLM 评分。

### 5. Token VLA、高层规划与连续动作头

| 工作 | 动作/数据路径 | 边界 |
|---|---|---|
| [RT-2](../entities/paper-rt-2.md) | VLM 生成 256 离散动作 token（基于 RT-1 ~130k 轨迹后续路线） | 数值量化、解码延迟 |
| [Open X-Embodiment](../entities/paper-open-x-embodiment.md) | 1M+ 轨迹、60 数据集、22 本体的跨本体数据集合 | 数据项目不等于单一模型 |
| [OpenVLA](../entities/paper-openvla.md) | 7B Llama 2 + DINOv2/SigLIP，970k 轨迹、7-D 离散动作 + LoRA | 归一化、模型计算、逐 token 解码 |
| FAST | DCT + 取整 + BPE 压缩高频动作块 | 压缩误差与高频接触细节 |
| OpenVLA-OFT | 并行解码、动作块、连续动作头、L1 回归 | 微调配方与特定任务协议 |
| [MolmoAct](../entities/paper-molmoact.md) / MolmoAct2 | depth tokens → 2D spatial trace → 动作 tokens（MolmoAct2 改用连续动作头） | 推理延迟与中间表征误差 |

```mermaid
flowchart LR
  X["图像与语言"] --> V["视觉语言骨干"]
  V --> T["离散自回归动作 tokens"]
  V --> H["并行连续动作头"]
  V --> M["深度 tokens"]
  M --> Q["空间轨迹"]
  Q --> T
  T --> U["解码与反归一化"]
  H --> S["动作块执行调度"]
  U --> S
```

图为多种独立模型的输出路径对照，不代表同一模型同时运行所有分支。高层规划方面，官方课件以 **Gemini Robotics 2（`ER 2`）** 为例展示由大模型拆解任务并调用低层 VLA 技能（而非二手笔记笼统提及的 GPT 系列）；直接调用大模型输出关节目标仍需校验动作、时限和安全接口。

### 6. Diffusion & Flow Policies 与 $\pi$ / GR00T 家族

连续动作生成主线通过多步去噪或速度场回归表达多模态连续分布（[Diffusion Policy](../methods/diffusion-policy.md)）：

- **Diffusion Policy（RSS 2023）：** 最小化噪声预测损失 $\mathcal{L}(\theta)=\mathbb{E}_{k,\epsilon}\|\epsilon-\epsilon_\theta(a_0+\sigma_k\epsilon,k,o)\|^2$；经典参数为预测地平线 $H=16$、执行步数 $E=8$、测试期 DDIM/Consistency 去噪步数 $K=10\text{–}100$。
- **Flow Matching：** 通过线性插值 $X_\tau=(1-\tau)\epsilon+\tau A$ 回归速度场 $\mathcal{L}=\mathbb{E}\|v_\theta(X_\tau,\tau,o)-(A-\epsilon)\|^2$，部署通常仅需约 $10$ 步。
- **代表架构对照：** **Octo**（RSS 2024，非 VLM，800k 轨迹/25 数据集，27M/93M）；**$\pi_0$**（PaliGemma 3B VLM + ~300M flow action expert，50 步动作块，`openpi`）；**$\pi$ 家族演进**为 $\pi_0 \to \pi_{0.5}$（子任务分解与新家庭泛化）$\to \pi_{0.6}$（更大 VLM，FAST+flow）$\to \pi^*_{0.6}$（Recap 从纠错与真机经验做 RL 微调）$\to \pi_{0.7}$（可调控技能组合）；**GR00T N1 / N1.6（开源 3B）** 将 System 2 VLM 与独立 System 1 DiT flow head 分离，区别于 $\pi_0$ 的共享注意力机制。

### 7. World Models 与系统集成选型（四角色与四层频率栈）

课件第 6–7 节（[World Models](../methods/generative-world-models.md)）把世界模型划分为四类角色：Learned simulator（如 Dreamer V3）、Planner（如 V-JEPA 2-AC：1M+ 小时视频 + 62 小时机器人视频，$a^*=\arg\min_a\|P(z_t,a)-z_{\text{goal}}\|$）、Data maker（Cosmos 1–2.5 到 2026-05 的 Cosmos 3 Reasoner+Generator 双塔，Super 32B & Nano 8B；Genie 3）与 **World Action Model（WAM）**（联合建模 $p(\text{video},\text{actions})=p(\text{video})\cdot p(\text{actions}\mid\text{video})$，如 DreamZero 与 GR00T N2）。

```mermaid
flowchart TB
  H["高层规划器 Planner<br/>秒级 · §4"] -->|"子任务目标"| M["VLA / WAM 策略<br/>~5-50 Hz · §4-6"]
  M -->|"动作块 / 末端或关节参考"| L["全身 RL 控制器<br/>~50 Hz · §3"]
  L -->|"高频目标位置/速度/前馈"| P["底层 PD 闭环<br/>~1 kHz · §1"]
```

- **按“缺失原料”选型（Slide 68）：** 仿真器+奖励 → PPO+DR；仿真器+动作片段 → AMP / Eureka / ROOT；单本体示范 → ACT / Diffusion Policy；跨本体多示范 → 微调 VLA（OpenVLA / $\pi_0$ / GR00T）；已部署策略+成功信号 → RL 微调（Recap）；海量视频+少量标签 → World Models / WAM。
- **抓红杯打滑五问排查表（Slide 70）：** 若怀疑增益/夹持力不足（§1）测接触力矩；怀疑示范无打滑样本（§2）查人工干预率；怀疑奖励忽略夹持（§3）对照奖励与失败轨迹；怀疑离散精度或延迟（§4–5）测分桶/chunk/延迟；怀疑想象物理摩擦失真（§6）对比预测视频与真机视频。

## 工程实践：动作块与反馈调度

设预测长度 $H$，执行前缀 $E$，控制指令周期 $\Delta t$，推理时延 $L$。每次只执行 $E$ 步再更新观测时，名义动作块覆盖 $H\Delta t$，重规划周期还受推理和调度影响。同步执行可在动作耗尽后等待；异步执行可一边运行旧块一边根据观测预测新块，需对齐新预测起点和已经执行的动作，防止过期动作。

```mermaid
sequenceDiagram
  participant O as 观测采集
  participant P as 策略
  participant B as 动作缓冲
  participant C as 低层控制
  O->>P: 观测与采集时间
  P->>B: H 步动作及目标时刻
  loop 执行 E 步
    B->>C: 当前目标
    C-->>O: 状态反馈
  end
  O->>P: 新观测
  P->>B: 更新块并对齐时间
```

该图为概念级同步滚动执行示意，并非某个官方仓库源码时序。异步方案需另查 [Action Chunking](../methods/action-chunking.md) 与 [控制重规划](../concepts/receding-horizon-policy-execution.md)。

建议对同一任务扫描 $H,E$，记录首次动作可用延迟、观测到执行延迟、执行缓冲欠载、动作跳变、任务成功率、任务完成时间与安全事件。不要把 chunk 越长或单次 forward 越快等同于整体闭环越好。

## 局限与风险

- 本页已完整对照 [ASU CSE 598（Fall 2026）71 页官方课件](../../sources/courses/asu_cse598_fall2026_physical_ai_robot_policies.md) 通读核对，但课件第 71 页注明“Sources for every slide are in the speaker notes”，PDF 正文提取未涵盖讲者备注及 YouTube 录像原声。
- 原先中文二手笔记关于“GPT 系列作为 planner”的表述与官方课件第 46 页展示的 Gemini Robotics 2（`ER 2`）不一致，本页已按课件原文更正。
- 该课程总览包含多个独立方法族，不包含单一统一官方可运行项目；源码运行时序图不适用，各论文/模型入口应在对应实体页中独立核对。
- 未对全部列出的原始论文逐篇复现实验，不将本页当作代替单篇论文深读。

## 关联页面

- [模仿学习](../methods/imitation-learning.md)
- [强化学习](../methods/reinforcement-learning.md)
- [VLA](../methods/vla.md)
- [Action Chunking](../methods/action-chunking.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [生成式世界模型](../methods/generative-world-models.md)
- [Sim2Real](../concepts/sim2real.md)
- [五大机器人学习范式](../comparisons/robot-learning-five-paradigms-taxonomy.md)
- [VLA 纵深路线](../../roadmap/depth-vla.md)
- [模仿学习纵深路线](../../roadmap/depth-imitation-learning.md)

## 推荐继续阅读

- [ACT 原始项目](https://tonyzhaozh.github.io/aloha/)
- [PPO 官方算法文档](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [FAST 官方 tokenizer](https://huggingface.co/physical-intelligence/fast)
- [OpenVLA-OFT 原始项目](https://openvla-oft.github.io/)
- [Awesome-Frontier-Model-Robotics（课件引用）](https://github.com/LENS-Lab-AI/Awesome-Frontier-Model-Robotics)

## 参考来源

- [ASU CSE 598（Fall 2026）Physical AI 官方课件归档](../../sources/courses/asu_cse598_fall2026_physical_ai_robot_policies.md)
- [human five 课程笔记归档及相关一手资料核对清单](../../sources/blogs/wechat_human_five_robot_policies_2026-11-24.md)
- [ACT 原始论文归档](../../sources/papers/act_arxiv_2304_13705.md)
- [RT-2 原始论文归档](../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md)
- [OpenVLA 原始论文归档](../../sources/papers/openvla_reading_arxiv_2406_09246.md)
