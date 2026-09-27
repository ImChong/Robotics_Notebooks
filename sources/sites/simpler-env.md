# simpler-env.github.io（SIMPLER / SimplerEnv 项目页）

- **标题：** Evaluating Real-World Robot Manipulation Policies in Simulation（SIMPLER）
- **类型：** site / project-page / benchmark
- **URL：** <https://simpler-env.github.io/>
- **arXiv：** <https://arxiv.org/abs/2405.05941>
- **代码：** <https://github.com/simpler-env/SimplerEnv>
- **机构：** UC San Diego、Stanford、UC Berkeley、Google DeepMind（多机构）
- **入库日期：** 2026-09-27

## 一句话摘要

**Real-to-sim** 评测：在 purpose-built 仿真里评估 **真实数据上训练** 的通才操作策略，通过控制/视觉对齐（系统辨识、绿幕背景、纹理烘焙等）使 **仿真成功率与真机强相关**（Google Robot + Bridge/WidowX 等 ~1500 episodes）。

## 开源状态（步骤 2.5，2026-09-27）

| 资源 | 状态 |
|------|------|
| arXiv / 项目页 | **已发布** |
| 评测代码与环境 | **已开源** — [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv)（Apache-2.0） |
| Colab 示例 | 项目页链到 `example.ipynb` |

**结论：** **已开源**（评测栈）；训练权重沿用 RT-1 / Octo 等外部 checkpoint。

## 关联资料

- 论文归档：[`simplerenv_arxiv_2405_05941.md`](../papers/simplerenv_arxiv_2405_05941.md)
- 仓库归档：[`simplerenv.md`](../repos/simplerenv.md)
- 沉淀实体：[`wiki/entities/paper-simplerenv-real2sim-eval.md`](../../wiki/entities/paper-simplerenv-real2sim-eval.md)
