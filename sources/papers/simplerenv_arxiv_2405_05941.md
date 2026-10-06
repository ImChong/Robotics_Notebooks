# Evaluating Real-World Robot Manipulation Policies in Simulation（arXiv:2405.05941 · SIMPLER）

> 来源归档（ingest · arXiv + 项目页 + GitHub）

- **标题：** Evaluating Real-World Robot Manipulation Policies in Simulation
- **类型：** paper / vla / evaluation / sim2real / manipulation
- **arXiv：** <https://arxiv.org/abs/2405.05941>
- **项目页：** <https://simpler-env.github.io/>
- **代码：** <https://github.com/simpler-env/SimplerEnv> — 归档见 [`sources/repos/simplerenv.md`](../repos/simplerenv.md)
- **机构：** 加州大学圣地亚哥分校（UCSD）；斯坦福大学（Stanford）；加州大学伯克利分校（UC Berkeley）；Google DeepMind
- **入库日期：** 2026-09-27
- **一句话说明：** 提出 **SIMPLER** 仿真评测套件：不做全保真 digital twin，通过对齐控制与视觉 gap，使 RT-1-X / Octo 等在 **Google Robot** 与 **Bridge V2 / WidowX** 设定上的 sim 分数与真机 **强相关**，并反映分布偏移敏感性。

## 开源状态（步骤 2.5）

- **核查日：** 2026-09-27。
- **已发布：** arXiv、开源环境、RT-1/Octo 等推理评测脚本与 Colab。
- **结论：** **已开源**（评测框架）。

## 摘录 1：问题——通才策略的可扩展评测

真实世界评测通才操作策略 **贵、难复现**；能力越广，faithful evaluation 负担越大。本文主张 **simulation-based evaluation** 作为 gold-standard 真机评测的 **可扩展代理**（与 sim-to-real **训练** 方向相反：这里是 **real-to-sim eval**）。

**对 wiki 的映射：** 升格 [`paper-simplerenv-real2sim-eval`](../../wiki/entities/paper-simplerenv-real2sim-eval.md)；互链 [VLA](../../wiki/methods/vla.md)、[Sim2Real](../../wiki/concepts/sim2real.md)（概念上为逆方向应用）。

## 摘录 2：不必 digital twin

关键思想：仿真环境 **不必** 像素级复刻真机场景，只需 **足够真实** 使得策略 sim 表现与真机 **相关**。缓解手段包括：离线系统辨识、**green-screen** 观测（真机背景贴图）、物体 **texture baking** 等。

**对 wiki 的映射：** 与 [SimFoundry](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md) 等「建场景」路线对照——SIMPLER 优先 **相关性** 而非全场景重建。

## 摘录 3：SIMPLER 套件与实证

**SIMPLER** 覆盖 RT-1 系 **Google Robot** 与 **BridgeData V2** WidowX 等常见设定；单 line import + Gym API；对 RT-1-X、Octo 等做 **paired sim-and-real**，~1500 episodes，报告 **强 Pearson 相关**；sim 还能反映 **distribution shift 敏感性** 等行为模式。

**对 wiki 的映射：** 更新 [`painode-116-xsimplerenv`](../../wiki/entities/paper-simplerenv-real2sim-eval.md) 指向 canonical 论文实体；VLA 方法页 benchmark 引用。

## BibTeX

```bibtex
@article{li2024simpler,
  title={Evaluating Real-World Robot Manipulation Policies in Simulation},
  author={Li, Xuanlin and Hsu, Kyle and Gu, Jiayuan and others},
  journal={arXiv preprint arXiv:2405.05941},
  year={2024}
}
```
