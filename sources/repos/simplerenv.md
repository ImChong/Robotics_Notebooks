# SimplerEnv（simpler-env/SimplerEnv）

- **Title:** SIMPLER — Simulated Manipulation Policy Evaluation for Real Robot Setups
- **URL:** https://github.com/simpler-env/SimplerEnv
- **Paper:** arXiv:2405.05941 — [`sources/papers/simplerenv_arxiv_2405_05941.md`](../papers/simplerenv_arxiv_2405_05941.md)
- **Project page:** https://simpler-env.github.io/
- **Type:** evaluation / benchmark / real-to-sim
- **License:** Apache-2.0（以仓库 LICENSE 为准）
- **入库日期：** 2026-09-27

## 核心内容（README 归纳）

- **定位：** 在仿真中评估 **在真实数据上训练** 的 manipulation 策略；提供 **SIMPLER** 环境集合与 **RT-1 / RT-1-X / Octo** 等推理评测脚本。
- **环境：** Google Robot（RT 系评测设定）、Bridge V2 / WidowX 等；Gym 接口，意图 **一行 import** 接入。
- **文档/示例：** 项目页 Colab `example.ipynb`；仓库含创建新环境与评测新 policy 的 guide。

## 开源状态（步骤 2.5，2026-09-27）

- **结论：** **已开源** — 环境 + 评测 workflow；策略权重来自各基线原仓库。

## 对 wiki 的映射

- [`wiki/entities/paper-simplerenv-real2sim-eval.md`](../../wiki/entities/paper-simplerenv-real2sim-eval.md)
- [`wiki/entities/painode-116-xsimplerenv.md`](../../wiki/entities/painode-116-xsimplerenv.md)（策展索引 → canonical 论文页）
- [VLA](../../wiki/methods/vla.md)、[Sim2Real](../../wiki/concepts/sim2real.md)
