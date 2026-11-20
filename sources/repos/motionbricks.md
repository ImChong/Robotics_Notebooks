# NVlabs / motionbricks（MotionBricks 独立官方源码仓）

> 来源归档（2026-11-20 源码核查）

- **仓库：** <https://github.com/NVlabs/motionbricks>
- **项目页：** <https://nvlabs.github.io/motionbricks/>
- **论文：** <https://arxiv.org/abs/2604.24833>
- **类型：** repo
- **入库日期：** 2026-11-20
- **通读：** 已完成
- **核对范围：** 2026-11-20 克隆并检出 `NVlabs/motionbricks`，通读根目录 `README.md`（L1–200）、`docs/runtime_api.md`（L1–200）、`motionbricks/model/engine.py`（L1–100）及连续转向示例 `docs/examples/locomotion_loop.py`（L1–135 全文），核实 `MotionBricksEngine` 的 TensorRT plan 构建与 CUDA graph 捕获、`GeneratedMotionPlayer`（SOMA 骨架）与 `GeneratedDOFMotionPlayer`（Unitree G1 关节空间）的 `generate -> sample -> record_pose` 60Hz 异步闭环运行时接口。

## 代码架构与关键入口（实测源码）

- **推理引擎层（`motionbricks/model/engine.py`）：** `MotionBricksEngine` 继承 `InferenceEngine`，构造期校验 ONNX 契约 `self.contract.validate_tensor_specs(self.tensor_specs)` 并构建/加载 `.trt-cache` TensorRT 引擎（支持 CUDA graph 固化），对外提供 `generate_animation(context, target, output)`。
- **播放器与引导目标层（`motionbricks/runtime/`）：** 针对 SOMA 骨架与 Unitree G1 分别封装 `GeneratedMotionPlayer` / `GeneratedDOFMotionPlayer`，配合 `place_target_poses` / `place_target_dof` 与 `adjust_target_pose_speed` / `adjust_target_dof_speed` 实现在线转向与动作拼接。
- **连续运行示例（`docs/examples/locomotion_loop.py`）：** 以 `60 Hz` 主循环监控转向角变化（`TURN_THRESHOLD=15°`）或接近过渡帧（`player.time_to_transition <= delta_time`），触发 `player.generate(target_poses, settings)` 后逐帧 `player.sample(...)` 与 `player.record_pose(...)`。

## 对 wiki 的映射

- 方法主入口：[motionbricks](../../wiki/methods/motionbricks.md)
- 论文实体页：[paper-motionbricks](../../wiki/entities/paper-motionbricks.md)
- GR00T WBC 上游仓归档：[gr00t_wholebodycontrol](./gr00t_wholebodycontrol.md)
