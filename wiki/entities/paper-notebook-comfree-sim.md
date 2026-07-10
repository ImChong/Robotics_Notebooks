---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2603.12185"
related:
  - ../overview/paper-notebook-category-11-simulation-benchmark.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_comfree-sim.md
summary: "接触建模慢、难并行的老问题，根子在于经典 LCP/NCP 式求解把所有接触点拧成一个全局互补约束（接触处「要么速度为零、要么力为零」），必须整体迭代求解。ComFree-Sim 提出免互补的接触公式：在库仑摩擦锥的对偶锥里，用一次阻抗式（impedance-style）的预测—修正（prediction–correction）就能把每个接触冲量算成闭式解。这样接触在接触对之间解耦、并在摩擦锥面（cone facets）之间可分离，天然对应 GPU 上「一个接触 / 一个锥面 = 一个线程」的并行模式，于是运行时随接触数量近线性扩展。公式进一步推广为统一的 6D 接触模型（切向 + 扭转 + 滚动摩擦），并给出一个实用的对偶锥阻抗启发式。引擎对外兼容 MuJoCo API，底层基于 NVIDIA Warp。"
---

# ComFree-Sim

**ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：11_Simulation_Benchmark），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

接触建模慢、难并行的老问题，根子在于经典 LCP/NCP 式求解把所有接触点拧成一个全局互补约束（接触处「要么速度为零、要么力为零」），必须整体迭代求解。ComFree-Sim 提出免互补的接触公式：在库仑摩擦锥的对偶锥里，用一次阻抗式（impedance-style）的预测—修正（prediction–correction）就能把每个接触冲量算成闭式解。这样接触在接触对之间解耦、并在摩擦锥面（cone facets）之间可分离，天然对应 GPU 上「一个接触 / 一个锥面 = 一个线程」的并行模式，于是运行时随接触数量近线性扩展。公式进一步推广为统一的 6D 接触模型（切向 + 扭转 + 滚动摩擦），并给出一个实用的对偶锥阻抗启发式。引擎对外兼容 MuJoCo API，底层基于 NVIDIA Warp。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| Contact-Rich | - | 接触密集：抓取、灵巧操作、行走等大量接触切换的任务 |
| LCP / NCP | (Non)Linear Complementarity Problem | (非)线性互补问题，经典接触求解的数学形式 |
| Complementarity | 互补约束 | "接触速度与接触力不能同时非零"的约束，把所有接触耦合在一起 |
| Friction Cone | 库仑摩擦锥 | 合法接触力（法向 + 切向）所在的圆锥约束 |
| Dual Cone | 对偶锥 | 摩擦锥的对偶集合，ComFree 在其中做投影/修正 |
| Cone Facet | 摩擦锥面 | 把圆锥线性化后的若干面，每个面可独立并行处理 |
| 6D Contact | - | 同时建模切向、扭转、滚动三类摩擦的统一接触模型 |
| Warp | NVIDIA Warp | NVIDIA 的 Python GPU 核函数框架，本引擎的实现后端 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 解决什么问题

**问题陈述**：接触密集任务的仿真为什么慢？因为主流物理引擎（MuJoCo、Bullet 等）把接触当成一个**互补问题**来解——所有接触点之间相互耦合，必须用迭代器（PGS、CG、Newton 等）**全局**求一个约束优化。接触越多，耦合越强，迭代越慢，而且这种全局耦合**很难切成 GPU 上互不依赖的并行任务**。

这就带来两难：

## 核心机制

1. **建模贡献**：提出**免互补**的接触公式——在摩擦对偶锥里用一次**阻抗式预测—修正**把接触冲量写成**闭式解**，从根上去掉了传统接触求解的全局耦合；
2. **可扩展性贡献**：利用"接触对解耦 + 锥面可分离"的双重结构映射到 GPU，达到**接触数量近线性**的运行时扩展；
3. **模型贡献**：统一的 **6D 接触模型**（切向/扭转/滚动摩擦）+ 实用的对偶锥阻抗启发式；
4. **工程贡献**：开源 [comfree_warp](https://github.com/asu-iris/comfree_warp)，**兼容 MuJoCo API**、基于 **NVIDIA Warp**，提供抓取与并行吞吐基准，便于直接接入现有机器人学习管线。

方法拆解（深读笔记小节）：核心思想：去掉互补约束；对偶锥里的阻抗式预测—修正；双重解耦 → 天然 GPU 并行；统一 6D 接触模型；工程落地：MuJoCo API + NVIDIA Warp。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 11_Simulation_Benchmark |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine.html> |
| arXiv | <https://arxiv.org/abs/2603.12185> |
| 发表 | 2026-03-12 (arXiv，2026-03-14 修订) |
| 项目主页 | [irislab.tech/comfree-sim](https://irislab.tech/comfree-sim/) |
| 源码 | [asu-iris/comfree_warp](https://github.com/asu-iris/comfree_warp)（NVIDIA Warp，兼容 MuJoCo API） |
| 笔记阅读日期 | 2026-06-10 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../overview/paper-notebook-category-11-simulation-benchmark.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_comfree-sim.md](../../sources/papers/humanoid_pnb_comfree-sim.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine.html>
- 论文：<https://arxiv.org/abs/2603.12185>

## 推荐继续阅读

- [机器人论文阅读笔记：ComFree-Sim](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine.html)
