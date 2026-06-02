# Shape Your Body: Value Gradients for Multi-Embodiment Robot Design

> 来源归档（ingest）

- **标题：** Shape Your Body: Value Gradients for Multi-Embodiment Robot Design
- **类型：** paper（机器人共设计 / 多具身 RL / 价值函数梯度搜索）
- **状态：** Under Review（项目页与 PDF 标注；arXiv HTML 已公开）
- **原始链接：**
  - 项目页：<https://nico-bohlinger.github.io/shape-your-body/>
  - PDF（TU Darmstadt）：<https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/shape_your_body.pdf>
  - arXiv HTML：<https://arxiv.org/html/2606.00702v1>
  - arXiv abs（若索引）：<https://arxiv.org/abs/2606.00702>
- **作者：** Nico Bohlinger, Jan Peters
- **机构：** Technical University of Darmstadt；Robotics Institute Germany (RIG) / DFKI / hessian.AI
- **入库日期：** 2026-06-02
- **一句话说明：** 先在最多 **50** 台腿足机器人、**190–1177** 维连续设计空间上训练 **URMA** 多具身策略与 **direct-design critic**，再冻结价值函数，用 **Value-Gradient Design Search (VGDS)** 在软信赖域内沿价值梯度优化机体参数；单次训练约 **7–9 h** 后，每个新设计约 **1–2 min**，相对「每个初值重跑 RL 共设计」基线边际成本极低。

## 核心论文摘录（MVP）

### 1) 问题：共设计仍按「每个机器人一条 RL 环」计费

- **链接：** PDF / arXiv HTML §1
- **摘录要点：** 经典共设计是双层优化：外环改 embodiment，内环为每个候选训/调控制器。RL 共设计（Transform2Act、BodyGen、Stackelberg PPO 等）可在单次训练中共享经验，但**换机器人、设计空间或任务就要重跑整条环**。真实腿足机的设计变量（质量、惯量、几何、关节限位、执行器、PD、名义关节位等）轻松到**数百维**。多具身 generalist 策略已有进展，本文问：**能否把多具身价值函数摊销成可复用的设计模型？**
- **对 wiki 的映射：**
  - [Shape Your Body 实体页](../../wiki/entities/paper-shape-your-body-value-gradient-design.md) — 动机与相对 FEACRL / RL 共设计基线的定位。

### 2) 训练：URMA + direct-design critic + PPO

- **链接：** §3.1；附录 B
- **摘录要点：**
  - 设计向量 $f \in [-1,1]^{d_{\mathrm{design}}}$，物理 embodiment $e=\Phi(f)$；每 episode reset 采样 $f_i$，episode 内固定；性能课程 $c_t$ 扩大设计支撑至全空间。
  - **URMA**：关节 token + description $d_j$（轴、限位、增益、质量、几何等）+ attention 聚合；critic 另含 foot 观测。
  - **direct-design critic**：标准 URMA critic 中 $d_j$ 只进 attention key；本文让 encoder $g_\psi(o_j, d_j)$ **同时吃观测与描述**，使 embodiment 更直接影响价值预测 → **更强的设计梯度**；$K$ 个 value head 集成取均值。
  - 任务：与 URMA 系列一致的 **速度跟踪 locomotion**（RL-X + MJX）。
- **对 wiki 的映射：**
  - 同上 wiki 页 — 「训练阶段」Mermaid 与 URMA 改动要点。

### 3) VGDS：冻结 critic + 状态库 + 软信赖域

- **链接：** §3.2；项目页公式
- **摘录要点：**
  - 冻结策略与 critic；从全设计空间 rollout 收集状态库 $\mathcal{S}=\{s_1,\ldots,s_M\}$。
  - 目标：$\hat{J}_\lambda(f)=\frac{1}{M}\sum_m \bar{V}(s_m,\Phi(f)) - \lambda \frac{\|f-f_{\mathrm{ref}}\|_2^2}{d_{\mathrm{design}}}$（$f_{\mathrm{ref}}$ 可为名义 URDF）。
  - 更新：Adam 梯度上升 + 每维步长 clip $\delta_{\max}$ + clip 到 $[-1,1]$；避免无约束 critic 外推导致设计崩到边界。
  - **VGDS** 与 BO、CMA-ES、PSO、CEM、DE、ARS、TuRBO、**GC-PFO** 等**共用同一冻结 critic 作 surrogate**；GC-PFO 为最强无 VGDS 专属结构的梯度类基线。
- **对 wiki 的映射：**
  - 同上 — 「设计搜索」流程与公式。

### 4) 实验规模与结论

- **链接：** §4；附录 A/E/G
- **摘录要点：**
  - **50** 基座机器人：15 四足、31 双足/人形、4 六足；设计随机化含质量、惯量、几何、关节限位、PD、执行器属性等。
  - **单机器人设计**（Go2 **358** 维、MIT Humanoid **514** 维、Golem **688** 维）：VGDS 显著改进 $f_{\mathrm{init}}$，Go2/Humanoid 上匹配最强基线，Golem 上 VGDS 改进最大。
  - **RL 共设计对比**（Schaff2019、FEACRL、Transform2Act、BodyGen、Stackelberg PPO，均适配 URMA+PPO）：VGDS 终性能 **持平或略优**；RL 基线随初值数**线性**增训练时间，VGDS **训练一次、搜索 1–2 min/设计**（可批处理 critic）。
  - **跨机器人泛化**：按 morphology class 留出目标机，或 **50 机全集** 训练；全集训练时 Go2/Humanoid 可超过 $f_{\mathrm{ref}}$；Golem 上 **仅六足子集** 训练的 critic 更好（全集被四足/双足主导）。
  - **设计分析**：按 body part × 参数类型分组 $f^\star-f_{\mathrm{ref}}$ — Humanoid 上名义关节位与 PD、足尺寸；Golem 上 action scale 与 P/D；Go2 上后腿轴、足几何、前髋/小腿速度限等；**无单参数组可解释全部增益**。
  - 高大人形（如 Fourier GR1-T2）优化后常更矮、更宽、更紧凑，利于稳定。
- **对 wiki 的映射：**
  - 同上 — 实验表与设计分析小节。

### 5) 局限与相关工作锚点

- **链接：** §5；§2
- **摘录要点：**
  - 仅**固定拓扑**的连续参数；不能加减关节（URMA 无拓扑梯度）。
  - 依赖 critic 覆盖与精度；信赖域缓解外推，但强参考设计不总可得。
  - 实验在 **MuJoCo/MJX** 简单仿真；未验证可制造性与真机迁移。
  - 架构与训练分布建立在 **URMA** 与大规模多具身腿足 RL 线（[One Policy to Run Them All](https://proceedings.mlr.press/v270/bohlinger25a.html)、[Embodiment scaling laws](https://arxiv.org/abs/2509.02815) 等）。
- **对 wiki 的映射：**
  - 同上 — 局限与推荐继续阅读。

## 相关资料（交叉核对，非正文逐字摘录）

| 资料 | 链接 | 备注 |
|------|------|------|
| 交互演示站 | <https://nico-bohlinger.github.io/shape-your-body/> | 浏览器内 URMA 策略 + Reference/Co-Design 滑条；多机种与 VGDS 迭代可视化 |
| 项目站归档 | [shape-your-body-nico-bohlinger.md](../sites/shape-your-body-nico-bohlinger.md) | TL;DR、公式、50 机训练集、Citation |
| URMA 原始 | MLR CoRL 2025 [bohlinger25a](https://proceedings.mlr.press/v270/bohlinger25a.html) | 统一形态编码的 multi-embodiment locomotion |
| URMA 规模化 | arXiv:2509.02815（URMAv2、极端 ER、50 机千万级 embodiment） | 同一作者线的训练基础设施 |
| 代码 | 项目页标注 *Code (soon)* | 入库时未发布 |

## 当前提炼状态

- [x] PDF + arXiv HTML §1–6 与方法公式已摘录
- [x] 项目页交互说明与训练集规模已对齐
- [x] wiki 映射：`wiki/entities/paper-shape-your-body-value-gradient-design.md`
