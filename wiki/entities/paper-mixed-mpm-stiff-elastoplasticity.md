---
type: entity
tags: [paper, simulation, mpm, elastoplasticity, granular, snow, fluid, siggraph, nvidia, newton]
status: complete
updated: 2026-09-15
doi: "10.1145/3811345"
venue: SIGGRAPH 2026
related:
  - ./newton-physics.md
  - ./paper-dat-divide-and-truncate.md
  - ./paper-kamino.md
  - ../queries/simulation-physics-fidelity.md
  - ./genesis-world-10.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/mixed_mpm_siggraph_2026.md
  - ../../sources/sites/nvidia-mixed-mpm.md
summary: "Mixed MPM（SIGGRAPH 2026，DOI 10.1145/3811345）：混合速度–应力离散的 MPM 族，面向 CFL 步长下刚性弹粘塑性（沙、雪、流体至近不可压）；紧凑 stencil GPU 隐式求解，作为 Newton 一等模块并与刚体双向耦合。"
---

# Mixed MPM：刚性弹粘塑性混合物质点法

**Mixed Material Point Methods for Stiff Elastoplasticity**（Gilles Daviet，[DOI:10.1145/3811345](https://doi.org/10.1145/3811345)，SIGGRAPH 2026，[项目页](https://research.nvidia.com/labs/prl/mixed_mpm/)）提出一族 **混合速度–应力离散** 的 **MPM**，面向 **CFL 步长** 下的 **刚性弹粘塑性** 材料，直至 **近不可压极限**。在 [Daviet & Bertails-Descoubes 2016] 混合格式上扩展 **有限应变粘弹性** 与更一般 **流动法则**；隐式积分得到 **对称、良定** 优化问题与 **紧凑 stencil** 的 GPU 求解器。作为 [Newton](./newton-physics.md) **一等模块**，与刚体求解器 **双向耦合**（颗粒推回关节角色，腿式机器人可因地形调整步态）。

## 一句话定义

**用混合 MPM 在可接受步长下仿真沙、雪、刚性弹塑性甚至近不可压流体，并和 Newton 刚体后端双向推–拉。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPM | Material Point Method | 物质点法：粒子携带物质、网格求梯度 |
| CFL | Courant–Friedrichs–Lewy | 显式稳定步长条件；本文追求 CFL-rate 隐式步 |
| GPU | Graphics Processing Unit | 紧凑 stencil 的单卡大规模颗粒仿真 |
| FEM | Finite Element Method | 与 MPM 并列的连续介质离散 |
| RL | Reinforcement Learning | 颗粒地形 + 腿式策略的潜在训练后端 |
| DAT | Divide and Truncate | Newton 多物理接触框架，与 MPM 刚–软耦合互补 |

## 为什么重要

- **颗粒与刚性固体是机器人场景常客：** 沙地、雪面、碎屑、可变形包装与 **腿式/轮式** 机体共存；需要 **颗粒→刚体** 反作用而不只是 one-way 地形。
- **刚性弹塑性此前难实时：** 经典 MPM 在 **硬材料 / 近不可压** 上步长或 stencil 成本高；混合离散把 **49M 颗粒落城** 压到单 GPU **~4 s/frame** 量级（项目页演示）。
- **与 Newton 栈原生集成：** 不必另接第三方 MPM；与 [DAT](./paper-dat-divide-and-truncate.md)、MuJoCo Warp、[Kamino](./paper-kamino.md) 同属引擎选型空间。
- **离散可切换：** 多种速度–应力对；**trilinear 速度** 常在精度/吞吐上具竞争力——便于工程折中。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 英伟达（NVIDIA Research，PRL） |
| **会议** | SIGGRAPH 2026 |
| **材料** | 沙、雪、弹性固体、近不可压流体、混凝土式断裂 |
| **Newton** | **first-party module**；无独立 GitHub |
| **开源** | **已开源（经 Newton）** — Apache-2.0 主仓 + `mpm_*` 示例 |

## 核心原理

### 混合离散

- **基础：** [Daviet & Bertails-Descoubes 2016a] 混合 MPM — 速度场与应力场 **不同插值空间**。
- **扩展：** 有限应变 **粘弹性** + 更一般 **弹塑性流动法则** → 覆盖沙、雪、弹性梁扭转至 **近不可压流体**。
- **隐式积分：** 每步化为 **对称、良定** 优化；**紧凑 stencil** 利于 GPU 并行。

### 刚体双向耦合

| 模式 | 行为 |
|------|------|
| One-way | 刚体/角色不受颗粒反作用（地形仅碰撞几何） |
| **Two-way** | 颗粒对关节角色与障碍 **施力**；演示中腿式机器人 **调整步态** |
| 交互沙盘 | 实时双向耦合编辑 |

### 流程总览

```mermaid
flowchart LR
  RB[刚体求解器<br/>MuJoCo / Featherstone]
  MPM[Mixed MPM<br/>颗粒/雪/流体/弹塑性]
  COUP[双向耦合力交换]
  DAT[DAT 接触后处理]
  RB <--> COUP
  MPM <--> COUP
  COUP --> DAT
  DAT --> STATE[统一 State / 传感器]
```

## 实验与评测

| 项 | 项目页/归档口径 |
|----|----------------|
| **吞吐标杆** | **49M 颗粒** 沙体落城，单 GPU **约 4 s/frame**（紧凑 stencil 的直接收益） |
| **材料谱覆盖** | 沙（颗粒流）、雪（裂缝传播）、弹性固体、近不可压流体、刚性弹塑性断裂 |
| **离散对照** | 多组速度–应力对可切换；**trilinear 速度** 在精度/吞吐上常具竞争力 |
| **耦合对照** | one-way（地形仅作碰撞几何）vs **two-way**（颗粒反作用推回关节角色）——演示中腿式机器人因地形改步态 |
| **RL 实证** | **无** — 论文侧重图形与物理演示，未给可变形地面上的策略训练曲线或 sim2real 结果 |

- **读法：** 4 s/frame 是 **特定场景 + 特定卡** 的演示数字，不是可跨页横比的基准成绩；换颗粒数、材料参数或 GPU 都会变。要估自己的吞吐，跑 Newton `mpm_*` 示例实测。
- **缺口提醒：** 「能仿真」不等于「能训策略」。大规模腿式 RL on deformable terrain 的样本效率与 [Sim2Real](../concepts/sim2real.md) 迁移，本文没回答，须自行验证。

## 源码运行时序图

实现随 [Newton](https://github.com/newton-physics/newton) `SolverImplicitMPM` 分发，无独立复现仓库。**不适用**单独 sequenceDiagram；最短路径：`pip install "newton[examples]"` → 运行 `mpm_*` / 刚–颗粒耦合示例 → 观察 `Solver.step` 与刚体状态同步更新。

## 工程实践

| 步骤 | 做法 |
|------|------|
| 安装 | Newton `[examples]` extra；核对 GPU 驱动 ≥545 |
| 示例 | 仓库 `mpm_*`：雪、水、颗粒与机器人耦合 |
| 耦合 | 需要 **地形反作用改步态** 时开 **two-way**；纯视觉地形可用 one-way 省算力 |
| 离散 | 大规模场景可优先评估 **trilinear 速度** 对 |
| 接触 | 与 [DAT](./paper-dat-divide-and-truncate.md) 组合理解多物理 **无穿透** 管线 |

## 局限与风险

- **无独立 repo：** 算法变更需跟踪 Newton 版本与 release note。
- **与 Genesis / 其他 MPM：** [Genesis](./genesis-world-10.md) 等亦支持 MPM；Newton 侧优势在 **与 MuJoCo/Kamino 同栈双向耦合**。
- **RL 实证：** 论文侧重图形/物理演示；大规模腿式 RL on deformable terrain 仍须自行验证样本效率与 sim2real。

## 与其他工作对比

| 路线 | 擅长材料 | 刚体耦合 | 与 Mixed MPM |
|------|----------|----------|--------------|
| **Mixed MPM** | 沙/雪/近不可压流体/**刚性弹塑性** | Newton 内 **双向** | 本页 |
| 经典显式 MPM | 软颗粒、流体 | 多为 one-way | 刚性与近不可压材料上步长被 CFL 卡死或 stencil 成本爆炸——这正是「混合速度–应力离散」要解的 |
| [Genesis](./genesis-world-10.md) 等第三方 MPM | 覆盖面接近 | 依各自引擎 | 能力上有重叠；Newton 侧的差异在 **与 MuJoCo Warp / [Kamino](./paper-kamino.md) 同栈双向耦合**，不必跨引擎搬状态 |
| FEM 软体 | 连续弹性体 | 视实现 | 拓扑不变的软体 FEM 更省；一旦发生 **大流动、断裂、颗粒化**，网格就跟不上了 |
| [DAT](./paper-dat-divide-and-truncate.md) | 不解材料，只解接触 | 作为后处理服务所有后端 | **不是竞品，是搭档**：MPM 负责颗粒本构，DAT 负责它与刚体/壳相遇时的无穿透几何 |
| 高度图/解析地形模型 | 只是几何 | 无反作用 | 算力便宜得多；只有当 **颗粒反作用力矩会改变步态** 时，才值得上 MPM |

- **最关键的分歧点：** **要不要 two-way**。one-way 把颗粒当地形贴图，系统性 **低估反作用力矩**；腿式机器人在沙雪上的步态调整恰恰来自这一项。算力预算紧时先问「这个任务的成败取决于反作用吗」，再决定开不开。
- **选型顺序建议：** 先按 **材料谱** 选流动法则（勿一律默认雪参数），再按 **规模** 选速度–应力离散对（大场景从 trilinear 起步），最后按 **任务** 决定耦合模式。
- **读法：** 以上为 **路线级** 对照；逐项定量比较以 **原文 PDF** 与项目页为准（[参考来源](#参考来源)）。

## 关联页面

- [Newton Physics](./newton-physics.md)
- [DAT](./paper-dat-divide-and-truncate.md)
- [Kamino](./paper-kamino.md)
- [仿真物理保真（Query）](../queries/simulation-physics-fidelity.md)
- [Genesis World 1.0](./genesis-world-10.md)

## 结论

**总判：Mixed MPM 把 Newton 的颗粒后端从「沙盒特效」推到刚性弹塑性与双向刚体耦合，是腿式/操作机器人在可变形地面上做高保真仿真的关键模块之一。**

1. **材料谱先对齐任务：** 沙/雪/流体/硬弹塑性选型不同流动法则，勿一律默认雪参数。
2. **双向耦合是步态敏感场景的开关** —— one-way 低估反作用力矩。
3. **复现走 Newton `mpm_*`**，不要找独立 GitHub。
4. **trilinear 速度** 常是性价比高的默认离散起点。
5. **与 DAT 同读** 理解刚–软–颗粒几何接触。
6. **SIGGRAPH 2026 / 项目页视频** 含 49M 颗粒与交互沙盘，适合估吞吐上限。

## 参考来源

- [Mixed MPM 论文归档](../../sources/papers/mixed_mpm_siggraph_2026.md)
- [NVIDIA Mixed MPM 项目页归档](../../sources/sites/nvidia-mixed-mpm.md)

## 推荐继续阅读

- [NVIDIA PRL：Mixed MPM 项目页](https://research.nvidia.com/labs/prl/mixed_mpm/)
- [DOI:10.1145/3811345](https://doi.org/10.1145/3811345)
- [Newton ImplicitMPM 文档与示例](https://github.com/newton-physics/newton)
