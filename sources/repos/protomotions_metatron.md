# NVlabs/ProtoMotions — METATRON 训练、推理与 C++ 部署源码读解

- **仓库：** <https://github.com/NVlabs/ProtoMotions>
- **关联权重：** <https://huggingface.co/nvidia/Metatron-v1>
- **核对对象：** 官方仓库 `NVlabs/ProtoMotions`（含 `nvidia/Metatron-v1`）
- **核对日期：** 2026-11-14
- **通读：** 已完成
- **核对范围：** `scripts/download_metatron.py`、`data/pretrained_models/metatron/README.md`、`examples/experiments/metatron/{README.md,g1_student.py}`、`examples/inference/metatron.py`、`protomotions/{inference_agent.py,launch/inference.py}`、`deployment/{metatron_policy.py,export_to_onnx.py}`、`deployment/native/{docs/metatron_v1_native.md,examples/metatron_v1_deploy.cpp}`、`demos/metatron/README.md`。

## 1. 模型下载与 Python 仿真推理入口

1. **下载预训练权重：** 运行 `python scripts/download_metatron.py` 将 `nvidia/Metatron-v1` 拉取至 `data/pretrained_models/metatron/`。
2. **MuJoCo 交互式推理：**
   ```bash
   python protomotions/inference_agent.py \
       --simulator mujoco --num-envs 1 \
       --checkpoint data/pretrained_models/metatron/G1/inference_last.ckpt \
       --inference-config examples/inference/metatron.py
   ```
   - `protomotions/inference_agent.py` 先导入对应模拟器后端，再委派给 `protomotions/launch/inference.py:run()`。
   - `run()` 调用 `load_experiment_configs_from_checkpoint()` 恢复训练期配置，随后由 `derive_inference_configs()` 执行 `examples/inference/metatron.py` 中的 `apply_inference_overrides()`，内部通过 `configure_masked_mimic_task(..., "idle", getup=GetupRecoveryConfig())` 挂载交互式提示任务与跌倒起身配置。
   - 依次构建 `build_env_from_configs()`、`AgentClass(..., config=agent_config, env=env)`、`agent.setup(training=False)`、`agent.load(..., for_inference=True)`，并在默认分支调用 `agent.evaluator.simple_test_policy()`（传 `--full-eval` 时改为 `agent.evaluator.evaluate()`）。
   - 查看器支持按 `X` 键在 `idle`、目标行走（target walk）、速度转向（velocity steer）与右手触达（right-hand reach）模式间切换。

## 2. 开源训练配方（`examples/experiments/metatron/`）与论文差异

- **教师跟踪器（`g1_teacher.py` / `soma_teacher.py`）：** 采用 5 身体部位的混合 FSQ / PPO 动作跟踪配方（无 AMP 判别器）。
- **学生蒸馏（`g1_student.py` / `soma_student.py`）：** 球面条件 VAE 掩码蒸馏配方（`LATENT_DIM = 64`、`TOKEN_DIM = 1024`、`NUM_FUTURE_STEPS = 5`、`STATE_KEYS = ["fk_max_coords_obs", "reduced_coords_obs", "last_action"]`），通过 `--expert-model-path` 指定冻结的五部位跟踪器检查点。
- **代码与论文的关键对齐说明：** `examples/experiments/metatron/README.md` 与 `g1_student.py` 文档头明确指出，开源配方演示的是**单教师（single-teacher）**蒸馏流程；而论文中用于 Unitree G1 实机的发布基础策略整合了 4 类领域专家教师跟踪器（`BONES`、`AMASS+LAFAN`、`recovery`、`SAMP`）。

## 3. ONNX 导出与 C++ TensorRT 真机部署链路

- **确定性 Prior 导出（`deployment/metatron_policy.py` & `deployment/export_to_onnx.py`）：**
  - `MetatronPriorExport` 包装 `MaskedMimicModel`，通过将 `VAE_NOISE_KEY` 置零关闭隐空间随机采样，使导出图仅保留先验网络与解码器（特权教师编码器从不进入推理计算图）。
  - `export_to_onnx.py` 可将 `BakedAnchorFK` 烘焙到导出图中供真机/浏览器端使用，或单独调用 `export_fk_to_onnx.py` 生成独立 FK 模型。
- **C++ 原生部署（`deployment/native/docs/metatron_v1_native.md` & `examples/metatron_v1_deploy.cpp`）：**
  - 原生入口 `metatron_v1_deploy_cpp` 对接 39 路输入契约 `v5_raw_context_scene`，通过 TensorRT 编译并加载策略引擎 `model.graphsafe.fp32.plan` 与独立 FK 引擎 `fk.fp32.plan`。
  - `MetatronTask` 管理四大控制模式：`kIdle`、`kSteering`、`kTeleop`、`kTracking`，并在 `50 Hz` 主循环中调用 `runtime.run_cycle(cycle_start)` 驱动 MuJoCo 仿真后端或 Unitree G1 硬件后端（含 `unitree_sensor_probe` 硬件自检、Xbox 手柄、Pico 头显躯干高度与 Dex-3 灵巧手触发器/残差 IK 集成）。
