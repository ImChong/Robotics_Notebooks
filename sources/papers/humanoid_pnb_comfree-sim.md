# ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine.html>
- **分类：** 11_Simulation_Benchmark
- **arXiv：** <https://arxiv.org/abs/2603.12185>
- **入库日期：** 2026-07-10
- **一句话说明：** 接触建模慢、难并行的老问题，根子在于经典 LCP/NCP 式求解把所有接触点拧成一个全局互补约束（接触处「要么速度为零、要么力为零」），必须整体迭代求解。ComFree-Sim 提出免互补的接触公式：在库仑摩擦锥的对偶锥里，用一次阻抗式（impedance-style）的预测—修正（prediction–correction）就能把每个接触冲量算成闭式解。这样接触在接触对之间解耦、并在摩擦锥面（cone facets）之间可分离，天然对应 GPU 上「一个接触 / 一个锥面 = 一个线程」的并行模式，于是运行时随接触数量近线性扩展。公式进一步推广为统一的 6D 接触模型（切向 + 扭转 + 滚动摩擦），并给出一个实用的对偶锥阻抗启发式。引擎对外兼容 MuJoCo API，底层基于 NVIDIA Warp。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-comfree-sim](../../wiki/entities/paper-notebook-comfree-sim.md).

## 对 wiki 的映射

- [paper-notebook-comfree-sim](../../wiki/entities/paper-notebook-comfree-sim.md)
- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../../wiki/overview/paper-notebook-category-11-simulation-benchmark.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine/ComFree-Sim__GPU-Parallelized_Analytical_Contact_Physics_Engine.html>
- 论文：<https://arxiv.org/abs/2603.12185>
