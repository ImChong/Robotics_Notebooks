# mjbatch

> 来源归档

- **标题：** mjbatch
- **类型：** repo
- **作者：** Kevin Zakka（zakka@berkeley.edu）
- **代码：** <https://github.com/kevinzakka/mjbatch>
- **PyPI：** <https://pypi.org/project/mjbatch/>（`pip install mjbatch`）
- **Stars：** ~365（2026-09-13）
- **入库日期：** 2026-09-13
- **许可证：** Apache-2.0
- **依赖：** `mujoco==3.11.0`、`numpy`
- **Python：** >=3.10
- **一句话说明：** 在 CPU 上用 C++ 线程池并行运行数千路 MuJoCo 仿真；`bind` 暴露批量 `MjData` 字段视图，`expand` 支持逐实例 `MjModel` 参数与 `set_const` 重算派生常量。
- **沉淀到 wiki：** 是 → [`wiki/entities/mjbatch.md`](../../wiki/entities/mjbatch.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（Apache-2.0） |
| **代码** | <https://github.com/kevinzakka/mjbatch> |
| **包分发** | PyPI `mjbatch` v0.1.0（2026-09-10） |
| **可跑入口** | `uv run examples/<file>.py`；部分示例需 `uv sync --group examples`；带窗口示例支持 `--headless` |

无独立项目页；以 GitHub README 与 PyPI 为准。

## README 要点（2026-09-13）

- **定位：** 原生 MuJoCo Python 绑定之上的 **CPU 批量并行层**——释放 GIL 的 C++ 线程池步进多份 `MjData`，策略侧仍用 NumPy 向量化读写 `qpos` / `ctrl` 等。
- **核心 API：**
  - `Batch(model, num_sims=…)` — 默认线程数 = 逻辑 CPU 数
  - `batch.bind("qpos")` / `batch.bind("ctrl")` — 跨仿真实例的 live 数组视图
  - `batch.expand("geom_friction")` — 每实例模型参数；配合 `set_const` 更新派生量
  - `batch.step()` — 并行前向积分
- **示例（`examples/`）：** `hello.py`、`cartpole_swingup.py`（iLQR）、`cartpole_mpc.py`（predictive sampling）、`g1_flip.py`（G1 后空翻 receding-horizon iLQR）、`go1_joystick.py`（PPO 摇杆行走，README 称 M1 笔记本约一分钟可训）、`arm_throw.py`（CEM 机构–控制协同设计）、`rizon_inertia.py`（阻尼 Gauss–Newton 惯量辨识）。

## 为什么值得保留

MuJoCo 生态已有 [MJX](../../wiki/entities/mujoco-mjx.md)（JAX/GPU 可微批量）与 [MuJoCo Warp](../../wiki/entities/mujoco-warp.md)（NVIDIA GPU 高吞吐），但缺少 **「沿用官方 Python `mujoco` 绑定 + NumPy 控制器」** 的轻量 CPU 千路并行入口。mjbatch 填补该缝隙，且与同作者 [Mink](../../wiki/entities/mink-ik.md) 一样强调 **零 MJCF 转换成本**。

## 对 wiki 的映射

- [mjbatch](../../wiki/entities/mjbatch.md)
- [MuJoCo](../../wiki/entities/mujoco.md)
- [Mink](../../wiki/entities/mink-ik.md) — 同作者 MuJoCo 微分 IK 库
