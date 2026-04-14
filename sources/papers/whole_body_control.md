# whole_body_control

> 来源归档（ingest）

- **标题：** Whole-Body Control / TSID / HQP
- **类型：** paper
- **来源：** arXiv / journal / 开源仓库
- **入库日期：** 2026-04-08
- **最后更新：** 2026-04-14
- **一句话说明：** 聚焦任务空间控制、层级 QP 与动力学一致控制，用于支撑 WBC、TSID、HQP 等核心页面。

## 核心论文摘录（MVP）

### 1) Task Space Inverse Dynamics (Del Prete et al.)
- **链接：** <https://ieeexplore.ieee.org/document/6651572>
- **核心贡献：** 提出在接触约束下的 prioritized motion-force 控制框架，统一求解 $\ddot{q}$、$	au$ 与接触力。
- **对 wiki 的映射：**
  - [TSID](../../wiki/concepts/tsid.md)
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)

### 2) Hierarchical Quadratic Programming: Fast Online Humanoid-Robot Motion Generation (Escande et al., 2014)
- **链接：** <https://www.researchgate.net/publication/274504328_Hierarchical_quadratic_programming_Fast_online_humanoid-robot_motion_generation>
- **核心贡献：** 系统化 HQP 的实时求解框架，明确任务优先级在 humanoid 控制中的工程可行性。
- **对 wiki 的映射：**
  - [HQP](../../wiki/concepts/hqp.md)
  - [TSID](../../wiki/concepts/tsid.md)

### 3) Synthesis of Whole-Body Behaviors Through Hierarchical Control of Behavioral Primitives (Sentis & Khatib)
- **链接：** <https://khatib.stanford.edu/publications/pdfs/Sentis_2005_ICHR.pdf>
- **核心贡献：** 奠定 whole-body 行为层级控制范式，是后续 WBC 任务优先级思想的重要来源。
- **对 wiki 的映射：**
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)

### 4) Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control (Mastalli et al., 2020)
- **链接：** <https://arxiv.org/abs/1909.04947>
- **核心贡献：** 在多接触最优控制上提供高效求解器，为 WBC/MPC 的中高层规划提供工程工具。
- **对 wiki 的映射：**
  - [Crocoddyl](../../wiki/entities/crocoddyl.md)
  - [MPC 与 WBC 集成](../../wiki/concepts/mpc-wbc-integration.md)

### 5) Capturability-based Analysis and Control of Legged Locomotion (Koolen et al., 2012)
- **链接：** <https://journals.sagepub.com/doi/10.1177/0278364912452673>
- **核心贡献：** 提出 N-step Capturable 概念，将 1-step/N-step capture point 形式化为可行域分析，为双足平衡恢复提供严格的可行性判定准则。
- **关键结论：** 0-step CP = Divergent Component of Motion（DCM）；N-step CP 递归定义扩展了可恢复扰动范围
- **对 wiki 的映射：**
  - [Capture Point / DCM](../../wiki/concepts/capture-point-dcm.md)
  - [Balance Recovery](../../wiki/tasks/balance-recovery.md)
  - [LIP / ZMP](../../wiki/concepts/lip-zmp.md)

## 当前提炼状态

- [x] 已补 TSID / HQP / Sentis&Khatib / Crocoddyl / Koolen N-step CP 五条主线摘要
- [~] 后续补：加入”QP-WBC 与 NMPC-WBC”架构级对比
