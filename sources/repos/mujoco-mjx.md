# mujoco-mjx（MuJoCo XLA / MJX）

> 来源归档

- **标题：** MuJoCo XLA（MJX）
- **类型：** repo（主仓子目录 + 独立 PyPI 包）
- **来源：** Google DeepMind
- **链接：** https://github.com/google-deepmind/mujoco/tree/main/mjx
- **PyPI：** https://pypi.org/project/mujoco-mjx/
- **官方文档：** https://mujoco.readthedocs.io/en/stable/mjx.html
- **入库日期：** 2026-05-18
- **一句话说明：** 用 JAX 重实现的 MuJoCo 物理内核：`pip install mujoco-mjx` 后以 `from mujoco import mjx` 调用，与 MuJoCo API 对齐并跟踪主版本号，功能完备度以文档为准。
- **沉淀到 wiki：** 是 → [`wiki/entities/mujoco-mjx.md`](../wiki/entities/mujoco-mjx.md)

## 为什么值得保留

- 把 **同一套 MJCF 资产** 接到 **JAX 可微 / pmap / 设备网格** 上，是 **大规模并行 RL**、**基于梯度的控制** 与 **Brax 训练栈** 之间的关键桥梁。
- **版本号与 MuJoCo 本体对齐**（`major.minor.micro` 一致，必要时 `.postN`），便于和 C API / Python `mujoco` 绑定对照升级。

## 对 wiki 的映射

- [MuJoCo MJX（实体页）](../../wiki/entities/mujoco-mjx.md)
