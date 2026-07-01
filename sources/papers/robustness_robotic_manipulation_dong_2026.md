# Robustness of Robotic Manipulation: Foundations and Frontiers

> 来源归档（ingest）

- **标题：** Robustness of Robotic Manipulation: Foundations and Frontiers
- **类型：** paper（arXiv preprint）
- **机构：** Duke University；KTH Royal Institute of Technology；Stanford University；MIT；Organifarms
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2606.31494>
  - PDF：<https://arxiv.org/pdf/2606.31494>
- **入库日期：** 2026-07-01
- **一句话说明：** 首篇以 **操作鲁棒性** 为核心对象的系统综述：给出任务中心定义与 POMDP / 鲁棒控制双视角形式化，按感知–规划–控制–策略学习–硬件五模块梳理机制，并归纳经验与解析两类评测协议及开放问题。

## 核心论文摘录（MVP）

### 1) 问题：子领域各自谈 robustness，缺乏统一语言

- **链接：** arXiv:2606.31494 §1 Introduction
- **摘录要点：** 真机操作面临位姿部分可观测、摩擦/柔顺难建模、接触事件离散不可预测、任务实例跨 episode 变化大；人类/动物在不确定环境中仍可靠操作，而机器人 pick-and-place 在接触不确定下仍易失败。各子领域（感知、规划、控制、学习、硬件）对 robustness 表述不一，概念常隐含未形式化，阻碍跨领域积累与沟通。
- **对 wiki 的映射：**
  - [操作鲁棒性综述](../../wiki/entities/paper-robustness-robotic-manipulation-survey.md) — 综述动机与贡献定位。

### 2) 定义：四维上下文 + 三类挑战

- **链接：** §2 Definition；Fig. 1（梨抓取示例）
- **摘录要点：**
  - **通用定义：** robustness = 在指定 **挑战** 下，借助特定 **机制**，通过 **评测** 量化，实现给定 **目标** 的能力（relational，非绝对属性）。
  - **操作特化：** 目标 = 通过接触交互达到期望物体状态；挑战 = **认知不确定性**（可减）、**偶然不确定性**（不可消）、**episode 变异**（跨次固定、次间变化）；机制 = 容忍/缓解不确定性与变异、防失败或恢复。
  - 类比：暗室找开关 — episode 变异（不同房间开关位置）、认知不确定性（可摸墙精化）、偶然不确定性（传感器噪声、手滑）。
- **对 wiki 的映射：**
  - [操作鲁棒性综述](../../wiki/entities/paper-robustness-robotic-manipulation-survey.md) — 定义与三类挑战框架。

### 3) 形式化：统一 POMDP 与双视角鲁棒性能

- **链接：** §3 Formulation；Eq. (1)–(6)
- **摘录要点：**
  - 离散时间部分可观测随机控制：$x_{t+1}=f(x_t,u_t,w_t;\theta)$，$y_t=g(x_t,v_t;\theta)$；episode $\nu=(\theta,x_0,\mathcal{G})$。
  - **概率视角：** POMDP，$\Gamma(\pi)=\mathbb{E}_{\nu}[J_\nu(\pi)]$，策略从 belief $b_t$ 映射到动作。
  - **控制视角：** $w,v,\nu$ 为有界未知量，min-max 最坏情况代价；与期望可靠性互补，提供严格保证。
- **对 wiki 的映射：**
  - [操作鲁棒性综述](../../wiki/entities/paper-robustness-robotic-manipulation-survey.md) — 数学形式化节。

### 4) 两大原则 × 五模块机制图谱

- **链接：** §4 Principles；§5 Mechanisms；Table 1–2
- **摘录要点：**
  - **原则轴 1 — 不确定性与变异调节：** 减少认知不确定性（主动/交互感知、belief 规划、世界模型）vs 容忍偶然不确定性与 episode 变异（柔顺、不变性、closure、域随机化、数据多样性）。
  - **原则轴 2 — 失败管理：** 防失败（预测控制、保守目标、closure 抓取）vs 恢复/容忍局部失败（重抓取、形态适应、多模态冗余、DAgger/在线 RL）。
  - **五模块（Table 1）：** 感知（主动感知、表征不变性、多模态）；规划（belief 推理、funneling、鲁棒裕度、closure）；控制（主动柔顺、控制不变性、预测性 MPC）；策略学习（训练分布、架构归纳偏置、鲁棒目标、部署适应）；硬件（被动柔顺、粘附、形态适应）。
- **对 wiki 的映射：**
  - [操作鲁棒性综述](../../wiki/entities/paper-robustness-robotic-manipulation-survey.md) — 机制总览与 Mermaid 主干图。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 任务页交叉引用鲁棒性框架。

### 5) 评测：经验成功率 vs 解析质量度量

- **链接：** §6 Evaluation；Eq. (8)–(9)
- **摘录要点：**
  - **经验协议：** Monte Carlo 任务成功率（对齐 §3 形式化）；阶段式子目标评测（approach / grasp / lift 等）；语义 rubric（Kress-Gazit et al. 2024）。
  - **解析协议：** 力/形 closure 抓取质量；裕度 $\sigma_{\text{margin}}=\mathrm{dist}(x,\partial\mathcal{G})$ 与 safety tube；STL 时序鲁棒性分数；收敛/发散（Jacobian 特征值）衡量自稳定动力学。
  - **局限：** 现有 benchmark 很少把 robustness 作为显式评测轴；STL 等单位不可比。
- **对 wiki 的映射：**
  - [操作鲁棒性综述](../../wiki/entities/paper-robustness-robotic-manipulation-survey.md) — 评测协议归纳。

### 6) 讨论与开放问题

- **链接：** §7–8
- **摘录要点：**
  - **与相关概念区分：** robustness ≠ safety / stability / generalization；泛化可助鲁棒但既不必要也不充分。
  - **开放问题：** 统一鲁棒性 benchmark（仿真 vs 真机权衡）；解析 vs 经验范式融合（know-why + know-how）；向人类级鲁棒需终身经验、物理世界模型、具身共设计与快速适应的整合。
  - **跨域启示：** 足式对抗训练、生物粘附等可迁移至操作；操作独有难点是多体接触组合复杂性。
- **对 wiki 的映射：**
  - [操作鲁棒性综述](../../wiki/entities/paper-robustness-robotic-manipulation-survey.md) — 开放问题与未来方向。

## 当前提炼状态

- [x] 定义、形式化、双原则、五模块机制表、评测协议、开放问题已摘录到可维护粒度
- [ ] 正式期刊/会议录用信息待更新
- [ ] 若作者发布配套 benchmark 或代码，应单独 `sources/repos/` 索引并回链

## 对 wiki 的映射（汇总）

- [操作鲁棒性综述](../../wiki/entities/paper-robustness-robotic-manipulation-survey.md)
- [Manipulation](../../wiki/tasks/manipulation.md)
- [Impedance Control](../../wiki/concepts/impedance-control.md)
- [Domain Randomization](../../wiki/concepts/domain-randomization.md)
- [Query：在 RL 中利用触觉反馈提升操作鲁棒性](../../wiki/queries/tactile-feedback-in-rl.md)
