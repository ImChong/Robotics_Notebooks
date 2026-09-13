# Mixed Material Point Methods for Stiff Elastoplasticity（NVIDIA PRL 项目页）

> 来源归档（site）

- **标题：** Mixed Material Point Methods for Stiff Elastoplasticity
- **类型：** site
- **作者：** Gilles Daviet
- **机构：** NVIDIA Research（PRL）
- **链接：** https://research.nvidia.com/labs/prl/mixed_mpm/
- **DOI：** https://doi.org/10.1145/3811345
- **会议：** SIGGRAPH 2026（[日程](https://s2026.conference-schedule.org/presentation/?id=papers_828&sess=sess103)）
- **入库日期：** 2026-09-13
- **一句话说明：** 混合 MPM 族面向 CFL 步长下的刚性弹粘塑性材料（至近不可压极限）；紧凑 stencil + GPU 隐式求解；作为 Newton 一等模块与刚体双向耦合。
- **代码：** **已集成** [Newton](https://github.com/newton-physics/newton)（`SolverImplicitMPM` / mixed 变体）；项目页未列独立仓库
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-mixed-mpm-stiff-elastoplasticity.md`](../../wiki/entities/paper-mixed-mpm-stiff-elastoplasticity.md)

---

## 开源状态（步骤 2.5，截至 2026-09-13）

| 资源 | 状态 |
|------|------|
| 项目页 / 论文 | 公开摘要、视频与性能演示 |
| 独立 GitHub | **无** — 明示为 Newton **first-party module** |
| Newton 主仓 | **已开源**（Apache-2.0）— 颗粒/雪/流体/弹塑性示例见 `mpm_*` |

**结论：算法随 Newton 引擎开源分发，无单独复现仓库。**

## 页面要点摘录

- 49M 颗粒沙体落城：混合离散紧凑 stencil，单 GPU **4 s/frame**。
- 速度–应力离散对（如 trilinear 速度）在性能/精度上常具竞争力。
- **双向刚体耦合：** 颗粒推回关节角色与刚体障碍，腿式机器人可因地形调整步态（对比 one-way）。
- 支持近不可压流体与雪崩裂缝、混凝土 Armadillo 压裂等刚性弹塑性展示。
