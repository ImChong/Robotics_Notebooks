# Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions

> 来源归档（sun254667 Awesome 策展索引级）

- **列表：** [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real)
- **分组：** 31 Real2Sim from Video
- **编号：** 028/063
- **入库日期：** 2026-08-10
- **arXiv：** 2511.04665
- **出处：** arXiv 2025（v2）
- **论文：** <https://arxiv.org/abs/2511.04665>
- **项目页：** <https://real2sim-eval.github.io/>
- **代码：** <https://github.com/kywind/real2sim-eval>
- **最后核对：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 全文通读 `arXiv:2511.04665v2`（xArm 7 + D405/D455 + GELLO 平台、Scaniverse + SuperSplat + ICP/RANSAC + IRLS 颜色对齐 + Warp PhysTwin + LBS、Toy packing/Rope/Push-T 3 项任务及 Table I 相关系数/MMRV 与 200-episode 扩展实验）并核对 `kywind/real2sim-eval` 仓库（`experiments/eval_policy.py`、`sim/envs/env.py:BaseEnv`、`GSRenderer`、`PhysTwinDynamics`、`PolicyInferenceWrapper`）

- **Highlights（清单）：** Real-to-sim robot policy evaluation with Gaussian splatting simulation of soft-body interactions
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2511-04665-real-to-sim-robot-policy-evaluation-with-gaussia.md`](../../wiki/entities/paper-sa-2511-04665-real-to-sim-robot-policy-evaluation-with-gaussia.md)

---

## 开源边界（2026-11-21 复核）

| 组件 | 状态 |
|------|------|
| 论文 `arXiv:2511.04665v2` & 项目页 | 已公开；含 Table I（r=0.901–0.944, MMRV=0.076–0.174）及 5–30 FPS/GPU 吞吐基准 |
| `kywind/real2sim-eval` 源码 | 已开源（含 `experiments/eval_policy.py`、`sim/envs/env.py` 仿真闭环与 HuggingFace `gs-scans`/`phystwin-*` 资产入口） |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-sa-2511-04665-real-to-sim-robot-policy-evaluation-with-gaussia.md`](../../wiki/entities/paper-sa-2511-04665-real-to-sim-robot-policy-evaluation-with-gaussia.md)
- 列表实体：[`wiki/entities/awesome-real2sim2real.md`](../../wiki/entities/awesome-real2sim2real.md)
