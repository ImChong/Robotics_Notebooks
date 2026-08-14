# TSID（stack-of-tasks）

> 来源归档

- **标题：** TSID — Task Space Inverse Dynamics
- **类型：** repo
- **来源：** Stack of Tasks / LAAS-CNRS / INRIA（Andrea Del Prete, Justin Carpentier 等）
- **链接：** <https://github.com/stack-of-tasks/tsid>
- **许可：** BSD-2-Clause
- **文档：** 仓库 wiki；教学页 <https://andreadelprete.github.io/#teaching>
- **入库日期：** 2026-08-13
- **一句话说明：** 基于 Pinocchio 的优化型任务空间逆动力学库；用 **HQP / 加权 QP** 实现任务优先级，是显式零空间投影器在带不等式（接触、限位）时的工程替代。
- **沉淀到 wiki：** [`wiki/concepts/tsid.md`](../../wiki/concepts/tsid.md)、[`wiki/concepts/null-space-control.md`](../../wiki/concepts/null-space-control.md)

## 开源核查（2026-08-13）

| 项 | 结论 |
|----|------|
| GitHub | `stack-of-tasks/tsid`；C++；BSD-2-Clause；默认分支 `devel` |
| 安装 | `conda install tsid -c conda-forge`；robotpkg `robotpkg-py3*-tsid`；源码依赖 Pinocchio + eiquadprog |
| 可运行入口 | `exercises/`（机械臂 / 人形 / 四足 Python 例）；`demo/demo_romeo.py`；`script/test_formulation.py` |
| 无独立项目页 | 以 GitHub + wiki + Del Prete 教学页为准 |

**结论：已开源。** 本库 wiki 此前只外链 GitHub，未建 `sources/repos/` 归档。

## 与零空间投影的关系

TSID **不**在用户代码里手写 $N=I-J^+J$。优先级改由分层 QP 表达：上层最优值成为下层等式约束——这是 Kanoun 2011 / Escande 2014 对 Nakamura 任务优先级的不等式推广。7 轴臂仍可作为 TSID 的「末端任务 + 关节正则」最小例子，但官方 demo 更偏人形 Romeo。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [TSID 概念](../../wiki/concepts/tsid.md) | 方法页 |
| [Pinocchio](./pinocchio.md) | 动力学后端 |
| [HQP](../../wiki/concepts/hqp.md) | 求解结构 |
| [零空间控制](../../wiki/concepts/null-space-control.md) | 连续时间投影 vs QP 分层对照 |
