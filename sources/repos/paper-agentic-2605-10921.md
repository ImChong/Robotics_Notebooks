# OpenHelix-Team/RoboMemArena

- **仓库地址：** https://github.com/OpenHelix-Team/RoboMemArena
- **关联论文：** [RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark（arXiv:2605.10921）](paper-agentic-2605-10921.md)
- **默认分支：** `main`
- **开源协议：** 仓库根元数据未声明全局许可证（仅 `baselines/astra_codex/LICENSE` 包含该基线适配器子目录的独立许可文件）
- **核对状态：** 静态代码与结构核对（基于仓库目录样本 `files_sample` 与官方 `README.md`；未在本地执行环境运行）

## 已核对目录与关键文件

- **数据集构建与格式转换：**
  - `RoboMemArena_dataset_builder.py`：将公开的 `subtask_data/`（按 `<primitive>_<subtask_order>_seed<seed>_task<task_id>.hdf5` 命名的关键帧分段 HDF5）或 ModelScope 上的 `full_trajectory/` 转换为 RLDS（TFDS）格式（依赖 Python 3.9、`tensorflow==2.13.0`、`tensorflow-datasets==4.9.2`、`h5py==3.9.0`、`numpy==1.24.3`）。
  - `conversion_utils.py`：数据集结构与格式转换辅助工具。
  - `subtask_data` HDF5 字段包含 `data/demo_{id}/actions`（`(T, 7)` 末端执行器动作）、`obs/agentview_rgb`（`(T, 256, 256, 3)` 顶视图像）、`obs/eye_in_hand_rgb`（`(T, 256, 256, 3)` 腕部相机图像）、`obs/ee_states`、`obs/gripper_states` 与 `obs/joint_states`。
- **任务定义（BDDL）：**
  - `bddl/` 与 `evaluation_benchmark/bddl/`：`1_cookies_tomato_basket.bddl` 至 `26_chocolate_pudding_cream.bddl` 共 26 个连续编号任务的 BDDL 场景与目标定义，另含 `distractor_name.txt` 与 `task_description.json`。
- **评测框架与参考运行入口：**
  - `evaluation_benchmark/libero_fork/`：配套的 LIBERO 兼容仿真分支（需加入 `PYTHONPATH` 并配置 `LIBERO_BDDL_PATH`）。
  - `evaluation_benchmark/async_vlm26_reference/eval_fullvlm26_async_vlm_vla.py` 与 `run_fullvlm26_async_vlm_vla_csr_tsr.sh`：异步双系统 VLM+VLA 参考评测入口，读取 `fullvlm_v2_26_memory_tasks.json` 并计算 CSR / TSR 指标。
  - `evaluation_benchmark/scripts/task2_26_reference_stage.py` 与 `evaluation_benchmark/docs/`（含 `evaluate_your_model.md`、`task_evaluation_code_guide.md`、`task11_contract.md`、`task22_stage_completion.md`、`counting_frypan_preconditions.md`）：分阶段状态谓词验证脚本与任务契约文档。
  - `third_party/openpi_minimal`（默认内置轻量运行时，亦可通过 `OPENPI_ROOT` 指向外部 OpenPI 仓库的 `scripts/serve_policy.py`）：低层 S1 策略服务接口。
  - `baselines/astra_codex/`：`run.py`、`codex_backend.py`、`robomemarena_adapter.py` 与单元测试 `tests/`，提供外部闭源/代码智能体基线适配器。
- **PrediMem S2 训练插件：**
  - `predictive_coding_head/`（见 `predictive_coding_head/README.md`）：在 Qwen3-VL 训练流程中接入 PrediMem S2 预测编码头的最小复用模块；S1 低层策略训练与环境复现直接沿用官方 `Physical-Intelligence/openpi`。

## 复现边界与工程注意点

- 2026 年 6 月 20 日仓库更新修复了 Task 6 的阶段判定逻辑（`evaluation_benchmark/scripts/task2_26_reference_stage.py`），提升了 Tasks 1–3 的双相同篮筐记忆依赖度，并刷新了 Tasks 4–5 的随机物体摆位数据；使用前需覆盖旧版缓存数据集。
- 本归档仅完成仓库结构、文档与入口文件清单的静态核对，未在本机拉起 MuJoCo/LIBERO 仿真或 OpenPI 推理服务。
- 对 wiki 的映射：[RoboMemArena 实体页](../../wiki/entities/paper-agentic-2605-10921.md)
