# FlashSAC（官方仓库）

> 来源归档

- **标题：** FlashSAC — Fast and Stable Off-Policy RL for High-Dimensional Robot Control
- **类型：** repo
- **代码：** <https://github.com/Holiday-Robot/FlashSAC>
- **论文：** <https://arxiv.org/abs/2604.04539>
- **项目页：** <https://holiday-robot.github.io/FlashSAC/>
- **入库日期：** 2026-07-16
- **一句话说明：** 论文官方实现：Hydra 配置 + `uv` 依赖管理；支持 **100+ 任务** 与 IsaacLab / MuJoCo Playground / ManiSkill / Genesis / HumanoidBench / MyoSuite / MuJoCo / Meta-World / DMC；含 checkpoint、IsaacLab `play_isaaclab.py` 可视化与 per-simulator 性能预设。
- **沉淀到 wiki：** [FlashSAC](../../wiki/methods/flashsac.md)

---

## 仓库要点（ingest 快照）

| 维度 | 内容 |
|------|------|
| **定位** | 高维机器人 off-policy RL 训练框架 + FlashSAC agent 实现 |
| **入口** | `train.py`（Hydra）；`play_isaaclab.py`（IsaacLab 可视化） |
| **配置** | `configs/flashSAC_base.yaml`；`configs/agent/`、`configs/env/` 模块化 |
| **仿真后端** | 默认 MuJoCo + DMC；可选 extras：`isaaclab`、`mujoco-playground`、`maniskill`、`genesis`、`humanoid-bench`、`myosuite`、`metaworld`、`all` |
| **工具链** | `uv sync`；Python 3.10（RTX 30/40）或 3.11（RTX 50/Blackwell）；`torch.compile` 模式随 Python 版本自动选择 |
| **日志** | Weights & Biases / TensorBoard |
| **目录** | `flash_rl/agents`、`buffers`、`envs`；`scripts/run_*.sh` 批量实验 |

## 性能预设（README 归纳）

| | GPU 仿真（IsaacLab、MJP、Genesis、ManiSkill） | CPU 仿真（MuJoCo、DMC、HBench、Myosuite） |
|---|---|---|
| `num_envs` | 1024 | 1 |
| `batch_size` | 2048 | 512 |
| AMP | On | Off |
| Buffer device | `cuda:0` | `cpu` |

## 依赖注意

- `mujoco-playground`：JAX > 0.5.2 有已知 NaN/崩溃问题；可能与 Python 3.11 不兼容。
- `isaaclab` 与 `genesis` / `humanoid-bench` **不能同环境安装**；需单独 venv。`all` extra **不含** `isaaclab`。

## 对 wiki 的映射

- [FlashSAC（方法页）](../../wiki/methods/flashsac.md)
- [sources/papers/flashsac_arxiv_2604_04539.md](../papers/flashsac_arxiv_2604_04539.md)
- [sources/sites/flashsac-project.md](../sites/flashsac-project.md)
