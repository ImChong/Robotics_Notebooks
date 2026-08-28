# Anytime Global Tensor Motion Planning

> 来源归档（ingest）

- **标题：** Anytime Global Tensor Motion Planning
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.25830>
- **作者：** Sai Coumar、An T. Le、Zachary Kingston（* 等贡献）
- **机构：** 普渡大学 CS；VinUniversity；TU Darmstadt IAS Lab
- **代码：** <https://github.com/CoMMALab/anytime_gtmp>（README 亦写 `commalab/anytime_gtmp`；GitHub 规范名为 `CoMMALab/anytime_gtmp`）
- **入库日期：** 2026-08-28
- **一句话说明：** 把 GTMP 的层状张量图接到任意黑盒局部规划器上；Anytime-GTMP 固定预算随机重启覆盖同伦类，AO-GTMP 知情扩张几乎必然收敛到最优代价。

## 核心摘录（MVP）

### 1) 一层采样图覆盖 δ-clear 同伦类

- **摘录要点：** 层间完全二分、层内无边。相邻层边由局部规划器（直线 SEV、样条、RRT-Connect、轨迹优化、生成采样）在查询椭球内实现。Thm.1：一层采样图以高概率覆盖每个有界长度、δ-clear 的端点固定同伦类；每层多样本指数降低 miss，更强局部器只亚线性减少所需层数。
- **对 wiki 的映射：**
  - [Anytime GTMP](../../wiki/entities/paper-anytime-gtmp.md)
  - [cuRobo](../../wiki/entities/curobo.md) — GPU 并行规划对照

### 2) 两条 anytime 策略

- **摘录要点：** Anytime-GTMP：固定 (M,N,s)，随机重启 + 按同伦类归档最低代价路径 → 几乎必然类覆盖（Thm.2）。AO-GTMP：固定 s、单调增大 (M,N)，在 informed set 里采样 → 几乎必然代价收敛（Thm.3，AO-x on DAG）。搜索是层状 DAG 上的张量 value iteration，O(M N² |K|)。
- **对 wiki 的映射：**
  - [Anytime GTMP](../../wiki/entities/paper-anytime-gtmp.md) — 流程与时序图
  - [Manipulation](../../wiki/tasks/manipulation.md)

### 3) MotionBenchMaker vs 2D 导航多样性

- **摘录要点：** 60 s 预算。Anytime-GTMP (SEV) 终态成功率约 **85%**，对齐 FCIT。AO-GTMP 在共同可解集上常拿到最低均路径代价（Panda 5/7、UR5 7/7、Fetch 5/7）。1 s 内不如 AORRTC/FCIT 快出第一条路径。2D 街道图上 Anytime-GTMP 覆盖最多同伦类；AO-GTMP / FCIT 会收缩到 1–2 类。局部器预算约 400–600 RRT-Connect 迭代后收益变平。
- **对 wiki 的映射：**
  - [Anytime GTMP](../../wiki/entities/paper-anytime-gtmp.md) — 评测读法

### 4) 开源状态（截至 2026-08-28）

- **摘录要点：** 论文写代码开源。仓库 MIT；`examples/benchmark_mbm_*.py` 与 `src/planners/` 可跑；依赖 `vamp`（`benchmark_aorrtc_backend`）与 `pyroffi` 子模块。
- **对 wiki 的映射：**
  - [Anytime GTMP 仓](../repos/anytime_gtmp.md)

## 当前提炼状态

- [x] arXiv HTML 方法 / 定理 / Table I 已对齐
- [x] GitHub README + examples 核查：**已开源**
- [x] wiki 映射：`wiki/entities/paper-anytime-gtmp.md`
