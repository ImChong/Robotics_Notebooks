# open_duck_playground

> 来源归档

- **标题：** Open Duck Playground
- **类型：** repo
- **来源：** apirrone（Open Duck Project）
- **链接：** https://github.com/apirrone/Open_Duck_Playground
- **Stars：** ~150+（2026-05）
- **入库日期：** 2026-05-28
- **一句话说明：** 基于 [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) 的 Open Duck 专用 RL 环境：JAX/MJX 并行训练、摇杆速度跟踪任务、Disney BDX 风格模仿奖励与 ONNX 导出，当前主力机型为 `open_duck_mini_v2`。
- **沉淀到 wiki：** 是 → [`wiki/entities/open-duck-playground.md`](../../wiki/entities/open-duck-playground.md)

---

## 技术栈

| 组件 | 说明 |
|------|------|
| 包管理 | **uv**（`curl -LsSf https://astral.sh/uv/install.sh \| sh`） |
| 仿真 | MuJoCo / **MJX**；MJCF 在 `playground/<robot>/xmls/` |
| 训练 | `playground/<robot>/runner.py`；共享 `playground/common/`（rewards、randomize、export_onnx） |
| 推理 | `mujoco_infer.py`（ONNX → MuJoCo）；TensorBoard 日志 |
| 上游灵感 | [kscalelabs/mujoco_playground](https://github.com/kscalelabs/mujoco_playground) |

## 目录结构（摘要）

```
playground/
├── common/          # export_onnx, rewards, randomize, runner, poly_reference_motion
└── open_duck_mini_v2/
    ├── base.py, constants.py, joystick.py, runner.py
    ├── data/polynomial_coefficients.pkl   # 模仿奖励用参考运动系数
    └── xmls/        # open_duck_mini_v2.xml, scene_mjx_*, assets/
```

## 训练任务与奖励

- **主任务：** `joystick` 环境 — 速度指令跟踪 + 可选域随机化（`randomize.py`）
- **模仿奖励：** 对齐 Disney [BDX 论文](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf) 的 imitation reward；需在 `joystick.py` 设 `USE_IMITATION_REWARD=True`，并将 [reference_motion_generator](open_duck_reference_motion_generator.md) 产出的 `polynomial_coefficients.pkl` 放入 `data/`
- **示例命令（flat + backlash）：**
  ```bash
  uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
  ```

## 与兄弟仓的接口

| 输入 | 来源 |
|------|------|
| MJCF / 轻量 config | [Open_Duck_Mini](open_duck_mini.md) + Onshape → `onshape-to-robot` |
| 电机参数 | [BAM](https://github.com/Rhoban/bam) → MJCF actuator 属性 |
| `polynomial_coefficients.pkl` | [Open_Duck_reference_motion_generator](open_duck_reference_motion_generator.md) |
| 真机部署 | 导出 ONNX → [Open_Duck_Mini_Runtime](open_duck_mini_runtime.md) |

## 扩展新机器人

复制 `open_duck_mini_v2` 目录 → 修改 `base.py`、`constants.py`（geom/sensor 命名）、`xmls/`、`joystick.py` 奖励、`runner.py`。

## 与本仓库 wiki 的映射

- 实体页：`wiki/entities/open-duck-playground.md`
- 交叉：[mjlab-playground.md](../../wiki/entities/mjlab-playground.md)、[brax.md](../../wiki/entities/brax.md)、[sim2real.md](../../wiki/concepts/sim2real.md)
