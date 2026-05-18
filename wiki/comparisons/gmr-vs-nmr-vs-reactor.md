---
type: comparison
tags: [motion-retargeting, humanoid, kinematics, reinforcement-learning, imitation-learning, bilevel-optimization, comparison, engineering-selection]
status: complete
updated: 2026-05-18
sources:
  - ../../sources/papers/motion_control_projects.md
  - ../../sources/papers/neural_motion_retargeting_nmr.md
  - ../../sources/papers/reactor_rl_physics_aware_motion_retargeting.md
related:
  - ../concepts/motion-retargeting.md
  - ../concepts/motion-retargeting-pipeline.md
  - ../formalizations/motion-retargeting-objective.md
  - ../methods/motion-retargeting-gmr.md
  - ../methods/neural-motion-retargeting-nmr.md
  - ../methods/reactor-physics-aware-motion-retargeting.md
  - ../methods/spider-physics-informed-dexterous-retargeting.md
  - ../methods/sonic-motion-tracking.md
summary: "GMR / NMR / ReActor 三种动作重定向方法谱系对比：监督优化（运动学 QP/IK）vs 学习式整段映射（神经网络 + 仿真锚定监督）vs 物理感知 RL（双层联合优化参考与跟踪策略），从输入形态、依赖、产物、训练 / 推理成本与适用场景给出选型坐标。"
---

# GMR vs NMR vs ReActor：动作重定向方法谱系对比

**背景**：当源动作（动捕、视频估计、生成模型）要喂给目标人形 / 异构机器人时，重定向是必须穿过的一道闸。围绕「**像不像**」与「**能不能跟得上**」两条评价线，社区涌现了三类代表性路线——以 **GMR** 为代表的**运动学优化**、以 **NMR** 为代表的**学习式整段映射**、以 **ReActor** 为代表的**物理感知 RL 双层优化**。三者并非互斥，工程上常组合使用，但在「数据来自哪里、误差在哪里修、推理预算多大」三个维度上的取舍泾渭分明。

> **一句话区分**：GMR 是「**先几何对齐、物理留给下游**」；NMR 是「**离线用 RL 把几何参考拉进可行流形，再训前向网络批量推断**」；ReActor 是「**在线把参考形变与跟踪策略放进同一个仿真双层优化里联合更新**」。

---

## 一句话定义

| 方法 | 一句话 | 论文 / 仓库 |
|------|--------|-------------|
| **GMR**（General Motion Retargeting） | 关键点 IK + QP 在运动学层做骨架对齐与限位平滑，CPU 实时输出关节轨迹。 | [arXiv:2505.02833](https://arxiv.org/abs/2505.02833) ；[YanjieZe/GMR](https://github.com/YanjieZe/GMR) |
| **NMR**（Neural Motion Retargeting） | 用 CEPR 数据管线（GMR 初值 → 聚类 → 每簇 PPO 专家跟踪 → 仿真 rollout）造**物理一致的人机配对**，训练 CNN–Transformer 做非自回归整段重定向。 | [arXiv:2603.22201](https://arxiv.org/abs/2603.22201) |
| **ReActor** | 把**参数化参考** \(\mathbf{p}\) 与**单一跟踪策略** \(\pi_\phi\) 写进双层目标，在物理仿真内交替更新；上层用结构化近似梯度，避免 Hessian 求逆。 | [arXiv:2605.06593](https://arxiv.org/abs/2605.06593) |

---

## 核心维度对比

| 维度 | **GMR**（运动学优化） | **NMR**（监督学习 + 仿真锚定） | **ReActor**（物理感知 RL 双层优化） |
|------|----------------------|--------------------------------|------------------------------------|
| **范式定位** | 单次几何 IK / QP 优化 | 离线构造配对数据 + 学习式整段映射 | 在线双层优化（参考 + 策略协同） |
| **典型输入** | MoCap / SMPL / 视频估计单帧或滑窗 | 整段 SMPL 序列 | 源动捕序列 + 稀疏语义刚体对应 + 根对应 |
| **输出形态** | 机器人关节轨迹 \(q_t\) | 机器人全身参考序列（整段并行） | 仿真内可执行参考 \(\mathbf{g}_t(\mathbf{p})\) + 跟踪策略 \(\pi_\phi\) |
| **物理一致性** | ❌ 仅运动学，下游须补动力学 | ✅ 数据级锚定到 RL 仿真可行流形 | ✅ 仿真内闭环，接触/自碰内生化 |
| **接触/自碰处理** | 启发式罚项 + 速度上限 | 通过 RL 专家的奖励内化 | 仿真接触求解器 + RFC 根残差力 |
| **训练成本** | 无（即开即用 QP） | 高：每簇 PPO + 网络两阶段训练 | 高：单环双层 + 大动作库 RL |
| **推理速度** | CPU 毫秒级 | GPU 前向 < 10ms / 段（非自回归） | 训练态产物；部署仍需独立跟踪策略 |
| **跨形态能力** | 重新配置骨架/关键点；与机型耦合 | 需重训 CEPR 与网络头 | 通过有界 \(\mathbf{p}\) 参数化跨人形/四足报告 |
| **对源噪声敏感性** | 高（脚滑/漂移会被刚性放大，见 ExoActor 反例） | 中（CEPR 物理筛选可吸收部分噪声） | 中–低（仿真闭环吸收，但受 \(\alpha\) 近似假设制约） |
| **核心假设** | 模型骨架与源骨架可在 IK 误差容忍内对齐 | RL 专家覆盖动作分布；TMR/聚类语义合理 | \(\partial \ell / \partial \mathbf{p}\) 可由策略响应标量化近似 |
| **是否需大网络/算力** | 否 | 是（CNN–Transformer + 多专家训练） | 否（无大网络，但需多环 RL 训练） |
| **典型代码产物** | `motion_retargeting.py` + URDF/MJCF 配置 | 预训练 + 微调 checkpoint + 推理脚本 | 训练得到的 \(\mathbf{p}^\*\) 参考 + 跟踪策略权重 |

---

## 数据流对比（Mermaid）

把三条路线放在同一张「**人体源动作 → 机器人参考 → 下游跟踪**」的坐标里，差异主要在**误差修补发生的位置**：

```mermaid
flowchart TD
  Src[人体源动作<br/>MoCap / 视频估计 / 生成]

  subgraph gmr_path["GMR：几何前端 + 下游兜底"]
    G1[关键点 IK / QP]
    G2[限位 + 速度上限 + 平滑]
  end

  subgraph nmr_path["NMR：CEPR 离线锚定 + 前向网络"]
    N1[GMR 初值]
    N2[硬阈值过滤<br/>速度/自碰/脚离地]
    N3[TMR 聚类 + 每簇 PPO 跟踪]
    N4[仿真 rollout → 配对数据集]
    N5[CNN–Transformer 训练<br/>预训练 + 微调]
  end

  subgraph reactor_path["ReActor：在线双层 RL"]
    R1[参数化参考 g_t(p)]
    R2[策略 π_φ + PD + RFC]
    R3[物理仿真<br/>接触 / 自碰 / 限位]
    R4[上层近似梯度 → 更新 p]
  end

  Src --> G1 --> G2 --> DownK[下游：动力学滤波 / RL tracking / WBC]
  Src --> N1 --> N2 --> N3 --> N4 --> N5 --> DownN[下游：WBC / RL tracking]
  Src --> R1 --> R2 --> R3 --> R4
  R3 -.-> R1
  R4 -.-> R2
  R3 --> DownR[产物即「参考 + 策略」]
```

要点：
- **GMR** 的误差修补点在**下游**（动力学滤波 / RL tracking）；
- **NMR** 把误差修补集中到**离线数据构造阶段**（仿真锚定为「物理真值」），换得推理侧的快速前向；
- **ReActor** 把误差修补放进**在线双层闭环**，参考与策略相互塑形。

---

## 适用场景

### 选 GMR 的场景

1. **源动作干净**（专业 MoCap / 标定良好的遥操作），人机骨架比例差异不大；
2. **需要 CPU 实时 / 低延迟**（在线遥操作、教师演示流式生成）；
3. **下游已有强 tracking**（如 BeyondMimic / SONIC），GMR 仅作前端覆盖；
4. **工程预算紧**：无需训练管线，几小时即可在新机型上跑通；
5. **作为更大流水线的零件**：NMR 自身仍依赖 GMR 做 CEPR 初值。

> **避坑**：源动作来自视频估计 / 生成模型时，GMR 的「刚性几何对齐」会放大全局漂移与脚滑——见 [ExoActor](../methods/exoactor.md) 中「跳过重定向直接喂 SONIC」的消融。

### 选 NMR 的场景

1. **存在大规模 SMPL / 同源人体库**（≥ 数千段），值得一次性把数据「拉进可行流形」；
2. **部署阶段需要毫秒级整段映射**（实时生成式控制、视频→人形流水线）；
3. **目标机型相对固定**（如论文以 Unitree G1 为主），可承担 CEPR + 网络训练沉没成本；
4. **下游策略希望吃到「时序一致、低伪影」的参考**以加速 IL/RL 收敛。

> **避坑**：CEPR 的「物理真值」由 RL 奖励与专家分簇决定，**分布外动作**（如非典型杂技、灵巧手细操作）仍可能被刚性映射到簇心附近；上肢精细 / 手部任务需单独评估。

### 选 ReActor 的场景

1. **强异构形态**（人形 ↔ 四足、不同肢长比例）下需要**统一**的参考生成框架；
2. **想把「参考形变」本身当作可优化对象**长期联合训练，而非「先做几何、再修动力学」的分阶段管道；
3. **接触/自碰约束希望内生化**而不是靠下游 QP 兜底；
4. **不需要前向重定向网络**——可以接受「训练态产物 = 参考 + 跟踪策略」一同部署。

> **避坑**：**RFC 根残差力**降低跨形态训练难度，但与「零辅助力、纯物理可复现」不是同一目标——阅读实验时需区分*参考质量*与*是否完全物理一致*；上层近似梯度依赖 \(\alpha\) 标量化策略响应，对分布外形态或奖励重塑需评估稳定性。

---

## 常见误判

1. **「NMR 取代 GMR」**：错。NMR 的 **CEPR 管线**显式以 GMR 输出为运动学初值，再用仿真 RL 做物理修补——是**叠加增强**而非替代关系。
2. **「ReActor 比 NMR 更先进」**：维度不同。ReActor 强调**在线双层优化**的算法贡献与跨形态适用性；NMR 强调**离线大规模配对 + 快速前向推断**。部署端是否需要毫秒级前向网络，往往是首要分水岭。
3. **「GMR 不物理一致 = 没用」**：GMR 在 MoCap → 机器人这种「源动作干净」的链路上仍是**收益项**，且是 NMR / SPIDER 等学习/采样路线的关键前端；其「非物理性」是分工问题，不是路线缺陷。
4. **「物理一致 = 上真机就稳」**：NMR / ReActor 的物理一致性是**仿真内**的，sim-to-real gap 仍会传导。真机部署仍需域随机化、SysID 校正与下游 WBC / RL tracking 的鲁棒性补强。
5. **「三者只能三选一」**：实际系统常**串联**：GMR 出运动学初值 → CEPR/ReActor 式仿真修补 → 下游 RL tracking。三条路线在「**误差修补位置**」这条轴上是连续谱，而非二元对立。

---

## 决策矩阵

```
你的主要约束是什么？
│
├── 源动作干净 + 部署要 CPU 实时 → GMR（必要时下游接动力学滤波）
├── 已有大规模 SMPL 库 + 部署要毫秒级前向 → NMR（CEPR + 神经网络）
├── 跨形态（人形/四足）联合参考生成 + 接触约束内生化 → ReActor
├── 工程预算紧 / 没有 GPU 集群 → GMR（再视下游 tracking 选型）
├── 想把「参考」当可微/可优化对象长期迭代 → ReActor
├── 源动作来自视频估计或生成模型，且有明显漂移 → 先评估「跳过重定向直接 tracking」（参考 ExoActor / SONIC 路线）
└── 灵巧手 / 接触歧义重的操作数据 → 考虑 SPIDER 类「采样优化 + 虚拟接触」前端
```

---

## 与其它对比页的区别

- 本页关注**三种重定向算法的谱系对比**（监督优化 / 学习式 / 物理感知 RL）；
- [RL vs IL](./rl-vs-il.md) 在**策略学习范式**层面对比，重定向只是数据预处理；
- [MPC vs RL](./mpc-vs-rl.md) 在**控制范式**层面对比，与重定向不在同一抽象层；
- [Sim2Real 方法横向对比](./sim2real-approaches.md) 关注的是「仿真→真机」的差距弥合，与本页「人体→机器人参考」是上下游关系。

---

## 参考来源

- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》对 GMR 的工程定位与局限点评。
- [sources/papers/neural_motion_retargeting_nmr.md](../../sources/papers/neural_motion_retargeting_nmr.md) — NMR / CEPR 管线与 CNN–Transformer 网络细节。
- [sources/papers/reactor_rl_physics_aware_motion_retargeting.md](../../sources/papers/reactor_rl_physics_aware_motion_retargeting.md) — ReActor 双层优化与近似梯度的算法叙事。
- Ze Y., et al. *GMR: General Motion Retargeting* — [arXiv:2505.02833](https://arxiv.org/abs/2505.02833)；技术报告 [arXiv:2510.02252](https://arxiv.org/abs/2510.02252)。
- NMR 项目主页：<https://nju3dv-humanoidgroup.github.io/nmr.github.io/> ；arXiv：<https://arxiv.org/abs/2603.22201>。
- ReActor arXiv 摘要页：<https://arxiv.org/abs/2605.06593> ；HTML 全文：<https://arxiv.org/html/2605.06593v1>。

---

## 关联页面

- [Motion Retargeting（动作重定向）](../concepts/motion-retargeting.md) — 任务定义与坐标系。
- [Motion Retargeting Pipeline（动作重定向流水线）](../concepts/motion-retargeting-pipeline.md) — 把三种方法放进同一条 8 阶段端到端管线的工程视角。
- [Motion Retargeting Objective（动作重定向目标函数形式化）](../formalizations/motion-retargeting-objective.md) — GMR / NMR / ReActor 在统一目标函数下的退化形态对照。
- [GMR（通用动作重定向）](../methods/motion-retargeting-gmr.md) — 几何前端方法细节。
- [NMR（神经运动重定向与人形全身控制）](../methods/neural-motion-retargeting-nmr.md) — CEPR 管线与网络结构。
- [ReActor（物理感知 RL 运动重定向）](../methods/reactor-physics-aware-motion-retargeting.md) — 双层优化与近似梯度。
- [SPIDER（物理感知采样式灵巧重定向）](../methods/spider-physics-informed-dexterous-retargeting.md) — 另一条「采样优化 + 虚拟接触」的物理感知前端。
- [SONIC（规模化运动跟踪）](../methods/sonic-motion-tracking.md) — 与「跳过重定向直接 tracking」路线对照。
- [ExoActor（视频生成驱动的人形控制）](../methods/exoactor.md) — 在估计/生成源动作上「何时跳过 GMR」的反例。

---

## 推荐继续阅读

- Peng X. B., et al. *DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills* — 角色动画侧「跟踪固定参考」的经典 RL 设定，与 ReActor「先造参考」对照。
- Müller M., et al. *ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting*（SIGGRAPH 2026 预印本） — 双层优化与跨形态报告。
- He T., et al. *Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation*（H2O / OmniH2O 系列） — 与 GMR + RL tracking 流水线衔接的工程参考。

---

## 一句话记忆

> **GMR 几何**、**NMR 监督**、**ReActor 物理**——三者按「误差修补发生在下游 / 离线 / 在线」分占谱系三端，工程系统往往不是三选一而是按需串联。
