# dynamicslab/hydrogym — 流体动力学强化学习平台

> 仓库来源归档（ingest）

- **类型：** repo / reinforcement-learning / cfd / flow-control / gymnasium / benchmark
- **URL：** <https://github.com/dynamicslab/hydrogym>
- **许可：** **MIT**
- **文档：** <https://dynamicslab.github.io/hydrogym>
- **数据集：** <https://huggingface.co/datasets/dynamicslab/HydroGym-environments>
- **Slack：** <https://join.slack.com/t/hydrogym/shared_invite/zt-27u914dfn-UFq3CkaxiLs8dwZ_fDkBuA>
- **入库日期：** 2026-09-02
- **一句话说明：** **Gymnasium 兼容** 的主动流控环境套件：**61+** 预配置流场、**6** 类 CFD 后端（Firedrake / MAIA LBM·FV / NEK5000 / JAX / JAX-Fluids），配套 Docker、HF checkpoint 与 SB3/RLlib 示例；Nature 2026 论文官方实现。

## 维护者整理的结构化入口（摘自 README）

| 主题 | 入口 |
|------|------|
| 快速开始（Docker 推荐） | README Quick Start；`clagemann/hydrogym-nvhpc-*` / `hydrogym-rocm-*` 镜像 |
| Firedrake 示例 | `examples/firedrake/getting_started/` |
| MAIA 示例（MPMD） | `examples/maia/getting_started/` |
| NEK5000 通道 / 翼型迁移 | `examples/nek/`（含 `2_channel`、`3_ppo` 零样本脚本） |
| JAX / JAX-Fluids | `examples/jax/`、`examples/jaxfluids/` |
| 流场配置目录 | `docs/FlowConfigurations.md` |
| Python 包 | `hydrogym/`：`core.py`、`FlowEnv`、`firedrake/`、`maia/`、`nek/`、`jax/`、`jaxfluids/` |

## 训练栈要点

- **环境构造：** `from hydrogym import FlowEnv` + 后端模块（如 `hydrogym.firedrake as hgym`）指定 `flow` / `solver` / `actuation_config`。
- **向量化：** `DummyVecEnv` + `VecNormalize` 与 Stable-Baselines3 **PPO** 等标准库对接。
- **Checkpoint：** 首次 `make_env` 时从 HuggingFace Hub **自动拉取** 网格；HPC 离线节点需预下载（见 `examples/maia/README.md`）。
- **高级：** 可微环境 + **GPPO**；**PettingZoo** 多智能体（NEK5000 3D 圆柱）；Paraview 导出、SLEPc 稳定性分析、modred DMD/POD。

## 对 wiki 的映射

- [`wiki/entities/paper-hydrogym.md`](../../wiki/entities/paper-hydrogym.md)
- [`wiki/entities/gymnasium.md`](../../wiki/entities/gymnasium.md)
- [`sources/papers/hydrogym_nature_s41586_026_10917_6.md`](../papers/hydrogym_nature_s41586_026_10917_6.md)

## 当前提炼状态

- [x] README 级入口、Docker/HF/后端分工已对齐
