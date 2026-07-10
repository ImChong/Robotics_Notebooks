---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2505.24068"
related:
  - ../overview/paper-notebook-category-10-sim-to-real.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_diffcotune.md
summary: "机器人控制器部署常受建模差异所困——为可计算而简化模型、或仿真器本身不准——通常需要临时手工调参才能在目标域达标。DiffCoTune 提出一个自动、基于梯度的调参框架，借助可微仿真器（differentiable simulators）提升部署域性能。方法迭代地采集 rollout，协同调（co-tune）仿真器参数与控制器参数，使迁移能在目标域少数几次试验内系统完成。具体地，构造多步目标并用交替优化有效把控制器适配到部署域。框架的可扩展性体现在：能对任意复杂度的模型法与学习法控制器协同调参，任务从低维倒立摆到高维四足与双足跟踪，在不同部署域均见性能提升。"
---

# DiffCoTune

**DiffCoTune: Differentiable Co-Tuning for Cross-domain Robot Control** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：10_Sim-to-Real），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

机器人控制器部署常受建模差异所困——为可计算而简化模型、或仿真器本身不准——通常需要临时手工调参才能在目标域达标。DiffCoTune 提出一个自动、基于梯度的调参框架，借助可微仿真器（differentiable simulators）提升部署域性能。方法迭代地采集 rollout，协同调（co-tune）仿真器参数与控制器参数，使迁移能在目标域少数几次试验内系统完成。具体地，构造多步目标并用交替优化有效把控制器适配到部署域。框架的可扩展性体现在：能对任意复杂度的模型法与学习法控制器协同调参，任务从低维倒立摆到高维四足与双足跟踪，在不同部署域均见性能提升。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Co-Tuning | 协同调参，同时调仿真与控制器参数 |
| Differentiable Sim | 可微仿真器，可对参数求梯度 |
| Cross-domain | 跨域，从一个域迁到另一个域 |
| Alternating Optimization | 交替优化，轮流优化两组参数 |
| Multi-step Objective | 多步目标，跨多个时间步的目标 |
| Rollout | 轨迹采样 |

## 为什么重要

- **"协同调仿真 + 控制器"胜过只调一边**：把 sim-to-real 当成双向适配问题；
- **可微仿真**让调参变成梯度优化，比随机域随机化更有方向性；
- **少试验迁移**对人形（真机试验昂贵危险）极具价值；
- 与本仓 10 模块其它迁移工作（MOSAIC、ZEST）互为不同路线。

## 解决什么问题

控制器部署的核心障碍是**建模差异**： - 为可计算而**简化模型**； - **仿真器不准**。

通常靠**手工临时调参**才能迁移，费时且不系统。DiffCoTune 要：**自动、基于梯度**地把控制器迁到目标域，且**少量试验**即可。

## 核心机制

1. **可微协同调参框架**：同时调仿真器与控制器参数，自动跨域迁移；
2. **多步目标 + 交替优化**：稳定有效地适配部署域；
3. **少试验迁移**：目标域几次 rollout 内完成；
4. **强可扩展**：模型法/学习法、低维到高维（四足/双足）通用。

方法拆解（深读笔记小节）：可微仿真 + 协同调参；多步目标 + 交替优化；迭代 rollout、少试验迁移；可扩展性；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 10_Sim-to-Real |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/DiffCoTune__Differentiable_Co-Tuning_for_Cross-domain_Robot_Control/DiffCoTune__Differentiable_Co-Tuning_for_Cross-domain_Robot_Control.html> |
| arXiv | <https://arxiv.org/abs/2505.24068> |
| 作者 | Lokesh Krishna、Sheng Cheng、Junheng Li、Naira Hovakimyan、Quan Nguyen（USC / UIUC） |
| 发表 | 2025 年 5 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-10-sim-to-real](../overview/paper-notebook-category-10-sim-to-real.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_diffcotune.md](../../sources/papers/humanoid_pnb_diffcotune.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/DiffCoTune__Differentiable_Co-Tuning_for_Cross-domain_Robot_Control/DiffCoTune__Differentiable_Co-Tuning_for_Cross-domain_Robot_Control.html>
- 论文：<https://arxiv.org/abs/2505.24068>

## 推荐继续阅读

- [机器人论文阅读笔记：DiffCoTune](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/DiffCoTune__Differentiable_Co-Tuning_for_Cross-domain_Robot_Control/DiffCoTune__Differentiable_Co-Tuning_for_Cross-domain_Robot_Control.html)
