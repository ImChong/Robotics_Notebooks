# Robo-Harness K1 官方代码仓归档（Robo-Harness/k1）

- **官方仓库：** https://github.com/Robo-Harness/k1
- **关联论文：** [Robo-Harness K1: Harnessing Robot-Use Agents via Perception Augmentation（arXiv:2609.29389）](paper-agentic-2609-29389.md)
- **开源协议：** MIT License（见仓库 `LICENSE` 与 `THIRD_PARTY_NOTICES.md`；外部依赖模型与仿真器保留各自原协议）
- **默认分支与版本：** `main`（当前对应 `v1.1`，含 2026-11-09 跨本体重构与 2026-11-23 实验更新）；另有归档分支/标签 `v1.0` 保留初版公开实现
- **开源状态判定：** **部分开源**（核心 harness 运行时、感知/控制工具接口、LIBERO-Pro 与 RoboSuite 适配器及 Qwen3.5 LoRA SFT 训练脚手架已随 MIT 协议公开；RoboCasa、RoboTwin、RoboDojo、RoboLab 适配器尚待公开发布；基准数据集、模型权重、原始轨迹日志、真机 Piper 控制器脚本及 SAM 3 / TAPNext++ 第三方权重需用户另行准备）
- **核对边界：** 静态代码核对，未本地运行（依据仓库 `files_sample`、配置文件清单与 `README.md` 核对模块划分与执行入口）

## 仓库结构与核心模块（基于实测文件清单）

- **包元数据与配置：**
  - `pyproject.toml`：发布包名 `robo-harness-k1`，Python 模块 `robo_harness`，命令行入口 `robo-harness` 与源码审计工具 `robo-harness-audit`。
  - `configs/libero-pro.yaml`、`configs/robosuite.yaml`、`configs/grasp-provider.example.yaml`：任务环境、感知后端（SAM 3 / TAPNext++ 或降级光流 `lk`）与抓取提供者配置模板。
  - `.env.example`：API 端点环境变量说明（`ROBO_HARNESS_BASE_URL`、`ROBO_HARNESS_MODEL`、`ROBO_HARNESS_API_KEY`、`ROBO_HARNESS_PROXY`）。
- **CLI 与运行时主循环：**
  - `src/robo_harness/cli.py`：命令行解析与 `robo-harness run --config ... --output ...`（支持 `--smoke` 单步空载测试）入口。
  - `src/robo_harness/runtime.py`、`src/robo_harness/deployment_config.py`、`src/robo_harness/transport.py`、`src/robo_harness/private_rpc.py`：OpenAI 兼容视觉-函数调用接口与感知后端 RPC 调度。
- **感知工具族与空间证据模块：**
  - `src/robo_harness/regions.py`、`src/robo_harness/roi.py`、`src/robo_harness/perception.py`：区域检索（SAM 3）、区域细查与跨视角一致性核验。
  - `src/robo_harness/geometry.py`、`src/robo_harness/geometry_receipt.py`、`src/robo_harness/scene_snapshot.py`、`src/robo_harness/self_geometry.py`：深度反投影、局部几何拟合、相机/机械臂自几何与结构化回执生成。
  - `src/robo_harness/tracked_points.py`：跨帧表面锚点维护与失效判定（支持 TAPNext++ 或 CPU 光流 `lk` 烟雾测试配置）。
  - `src/robo_harness/grasp_backends.py`、`src/robo_harness/grasp_geometry.py`、`src/robo_harness/grasp_observation.py`、`src/robo_harness/grasp_presentation.py`、`src/robo_harness/grasp_visualization.py`：抓取候选后端适配、指垫坐标映射与投影示意图渲染。
- **运动控制、执行契约与双臂扩展：**
  - `src/robo_harness/motion.py`、`src/robo_harness/motion_limits.py`、`src/robo_harness/point_targets.py`、`src/robo_harness/pose_targets.py`、`src/robo_harness/position_reach.py`、`src/robo_harness/waiting.py`：相对微调、绝对位姿目标、全目标直达与夹爪稳定等待。
  - `src/robo_harness/control_reference.py`、`src/robo_harness/reference_alignment.py`、`src/robo_harness/reference_tracking.py`、`src/robo_harness/bimanual.py`、`src/robo_harness/multi_effector.py`、`src/robo_harness/multi_effector_tools.py`：参考系对齐与单/双执行器统一工具接口。
  - `src/robo_harness/execution_contract.py`、`src/robo_harness/execution_ledger.py`、`src/robo_harness/in_process_execution.py`、`src/robo_harness/contracts.py`、`src/robo_harness/tool_surface.py`、`src/robo_harness/resource_errors.py`。
- **记忆、反馈、完成协议与审计：**
  - `src/robo_harness/episode_history.py`、`src/robo_harness/decision_trace.py`、`src/robo_harness/task_progress.py`、`src/robo_harness/output_budget.py`、`src/robo_harness/guidance.py`：滑动文本/图像窗口、历史检索、里程碑记录与输出截断恢复。
  - `src/robo_harness/completion_protocol.py`、`src/robo_harness/feedback_presentation.py`、`src/robo_harness/gripper_monitor.py`、`src/robo_harness/gripper_evidence.py`、`src/robo_harness/gripper_signals.py`、`src/robo_harness/interaction_evidence.py`、`src/robo_harness/phase_confirmation.py`、`src/robo_harness/model_video.py`。
  - `src/robo_harness/release_audit.py`：源码发布前密钥、私有路径与二进制资产扫描（`robo-harness-audit`）。
- **环境适配器与智能体蒸馏训练：**
  - `src/robo_harness/libero_adapter.py`、`src/robo_harness/robosuite_adapter.py`、`src/robo_harness/robosuite_tasks.py`：已公开发布的 LIBERO-Pro 与 RoboSuite 仿真器适配（README 建议为两套环境分别建立独立的 Python 3.10/3.11 虚拟环境）。
  - `src/robo_harness/training/__init__.py`、`src/robo_harness/training/prepare.py`、`src/robo_harness/training/sft.py`：从轨迹日志构建精确请求监督样本并执行 Qwen3.5 LoRA SFT。
- **测试套件：**
  - `tests/` 下共 34 个测试文件（如 `test_motion.py`、`test_grasp_backends.py`、`test_completion_protocol.py`、`test_episode_memory.py`、`test_training_loss.py` 等），README 明确其采用合成观测与 mock API 响应运行，不代表端到端基准复现或真机验证。

## 文档与映射

- 仓库文档清单：`docs/architecture.md`、`docs/results.md`、`docs/environments.md`、`docs/training.md`、`docs/validation.md`、`docs/releases/v1.1.md`、`docs/usage_zh.md`。
- 对 wiki 的映射：[Robo-Harness K1](../../wiki/entities/paper-agentic-2609-29389.md)
