# google-deepmind/mujoco_warp

> 来源归档

- **标题：** MuJoCo Warp（MJWarp）
- **类型：** repo
- **组织：** Google DeepMind + NVIDIA（Newton 项目一部分）
- **代码：** <https://github.com/google-deepmind/mujoco_warp>
- **文档：** <https://mujoco.readthedocs.io/en/latest/mjwarp/index.html>
- **Nightly 基准：** <https://google-deepmind.github.io/mujoco_warp/nightly/>
- **PyPI：** `mujoco-warp`
- **Stars：** ~1.4k（2026-09-05）
- **入库日期：** 2026-09-05
- **许可证：** Apache-2.0
- **一句话说明：** 面向 NVIDIA GPU 的 MuJoCo 高吞吐实现；Newton 的主要刚体后端，也被 mjlab / Isaac Lab `feature/newton` / Playground（经 MJX）使用。
- **沉淀到 wiki：** 是 → [`wiki/entities/mujoco-warp.md`](../../wiki/entities/mujoco-warp.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（Apache-2.0） |
| **代码** | <https://github.com/google-deepmind/mujoco_warp> |
| **安装** | `pip install mujoco-warp`；开发 `uv sync --all-extras` |
| **文档** | MuJoCo readthedocs `mjwarp/` |
| **可跑入口** | `python benchmarks/run.py -f unitree_g1_flat --view`；[Colab tutorial](https://colab.research.google.com/github/google-deepmind/mujoco_warp/blob/main/notebooks/tutorial.ipynb) |

README 写明：快仿真要 NVIDIA GPU；CPU 仅开发/调试。

## README 要点（2026-09-05）

- DeepMind 与 NVIDIA **作为 Newton 的一部分** 共同维护。
- 示例场景：`unitree_g1_flat` / `g1_hfield`、`myoarm`、`aloha_*`、`three_humanoids`、`cloth`；夜间公开基准。
- **接入路径：**
  - 多数情况可当 MuJoCo 的 drop-in。
  - JAX：经 [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) + [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)。
  - PyTorch：[Isaac Lab `feature/newton`](https://github.com/isaac-sim/IsaacLab/tree/feature/newton) 经 Newton；或 [mjlab](https://github.com/mujocolab/mjlab) 直接铺 manager API。
- **API 缺口（相对 CPU MuJoCo）：**
  - Integrator：`IMPLICITFAST` midpoint **不支持**
  - Solver：`PGS`、`noslip` **尚未支持**
  - Actuator / Sensors：`PLUGIN` **尚未支持**
  - Flex：实验性，未全实现/优化
- **可微：** 经 Warp 的可微 **尚不可用**（[issue #500](https://github.com/google-deepmind/mujoco_warp/issues/500)）。
- **批量渲染：** GPU 光线追踪，多世界多相机；网格 / 纹理 / heightfield / Flex / 高斯溅射 / 异构相机 / 光照阴影。

性能剖析：`mjwarp-testspeed ... --event_trace`。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| MJWarp 定位、缺口、渲染 | [`wiki/entities/mujoco-warp.md`](../../wiki/entities/mujoco-warp.md) |
| 计算底座 | [`wiki/entities/nvidia-warp.md`](../../wiki/entities/nvidia-warp.md) |
| 引擎 / 框架 | [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)、[`wiki/entities/mjlab.md`](../../wiki/entities/mjlab.md)、[`wiki/entities/mujoco-mjx.md`](../../wiki/entities/mujoco-mjx.md) |
