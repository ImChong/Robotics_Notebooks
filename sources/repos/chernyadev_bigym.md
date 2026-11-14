# chernyadev/bigym 官方代码仓库核对归档

> 来源归档（GitHub 源码实体核对）

- **仓库：** <https://github.com/chernyadev/bigym>
- **类型：** repo
- **许可证：** Apache-2.0（根目录 `LICENSE`）
- **核对范围：** 核对 `README.md`、`setup.py`、`CHANGELOG.md`、`LICENSE`、`bigym/const.py`、`bigym/bigym_env.py`、`bigym/action_modes.py`、`bigym/robots/configs/h1.py`、`demonstrations/demo_store.py`、`demonstrations/demo_player.py`、`demonstrations/demo_converter.py` 与 `examples/replay_demo.py`。
- **通读：** 已完成
- **入库日期：** 2026-11-14

## 源码架构与运行核对

1. **仿真依赖栈（`setup.py`）**：绑定 `mujoco==3.1.5`、`mojo==0.1.1`、`dm_control==1.0.19`、`gymnasium` 与 `safetensors`（演示权重/轨迹轻量化存储）。
2. **环境与频率常量（`bigym/const.py`、`bigym/bigym_env.py`）**：`PHYSICS_DT = 0.002`（500 Hz 物理步长），控制频率支持 `CONTROL_FREQUENCY_MIN = 20` 到 `CONTROL_FREQUENCY_MAX = 500` Hz；`BiGymEnv` 继承 `gymnasium.Env`，通过 `Mojo` 加载场景与 `H1`（含 Robotiq 2F-85 夹爪与腕部 hinge 自由度）。
3. **动作模式（`bigym/action_modes.py`）**：`PelvisDof` 定义浮动基座自由度（`X, Y, RZ` 默认）；提供 `JointPositionActionMode(floating_base=..., absolute=...)` 与 `TorqueActionMode`；`README.md` 明确提示当前公开演示数据集仅适配 `JointPositionActionMode(floating_base=True, absolute=True/False)`。
4. **轻量演示拉取与重放（`demonstrations/`、`examples/replay_demo.py`）**：`DemoStore` 从 `github.com/chernyadev/bigym_data` Releases（`v0.9.0`）按需下载 `.safetensors` 动作轨迹并本地缓存；`DemoConverter.decimate` 与 `absolute_to_delta` 负责频率下采样及动作模式转换；`DemoPlayer` 在 `BiGymEnv` 中回放动作以即时渲染 RGB-D 观测。`README.md` 同时提示并非全部原始演示都能通过当前校验。

## 对 wiki 的映射

- [paper-notebook-bigym-a-demo-driven-mobile-bi-manual-manipulatio](../../wiki/entities/paper-notebook-bigym-a-demo-driven-mobile-bi-manual-manipulatio.md)
