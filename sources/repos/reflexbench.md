# LxRoboticsLab/ReflexBench 官方仓库

> 来源归档（repo）

- **标题：** ReflexBench: Official Isaac Lab Implementation & Latency-Aware Evaluation Suite
- **类型：** repo
- **链接：** <https://github.com/LxRoboticsLab/ReflexBench>
- **关联数据集：** <https://huggingface.co/datasets/cyx337/ReflexBench_dataset>（LeRobot v3.0 格式，约 200 episodes/task × 6 tasks）
- **arXiv：** <https://arxiv.org/abs/2608.14379>
- **项目页：** <https://reflexvla.github.io/>
- **协议：** BSD 3-Clause License
- **入库日期：** 2026-11-14
- **一句话说明：** 论文 arXiv:2608.14379 配套开源的 Isaac Lab 动态操纵基准仓，包含 6 个反应关键任务环境、cuRobo/RL 数据采集及 LeRobot 转换管线，以及支持同步/异步/真实延迟（RTF 换算）的 HTTP 策略评测客户端，但**不包含 ReflexVLA 策略本体代码**。
- **沉淀到 wiki：** [`wiki/entities/paper-reflexvla.md`](../../wiki/entities/paper-reflexvla.md)
- **通读：** 已完成
- **核对范围：** 通读 `_tree.txt`（126 个文件清单）、`README.md`（L1-450 全文）、`LICENSE`（BSD 3-Clause），核对核心脚本与函数：`scripts/evaluation/eval.py`（`Evaluator` 类、`setup`、`_run_sync`、`_run_async`、`_run_async_realtime`、`_current_rtf`、`_real_latency_period_ms`、`main`）、`scripts/evaluation/policy_client.py`（`PolicyClient` 与 `/info` `/predict` `/reset` REST 协议）、`scripts/evaluation/POLICY_SERVER.md`、以及 `scripts/data_collection/collect_vla_data.py` 与 `planning_policy.py` 模块入口，并全仓检索确认是否含 ReflexVLA 模型源码。

## 开源状态

- **基准环境与数据已开源；ReflexVLA 策略本体未开源**（2026-11-14 全仓代码检索与文档核查）：
  - **已开源（BSD 3-Clause）：** 本仓库提供完整 6 个 Isaac Lab 动态操纵任务（`ConveyorBeltPickAndPlace`、`BallCatching`、`WhackAMole`、`RollingBallInterception`、`BallThrowing`、`RotatingPegInsertion`）、3 种动作空间变体（`joint_pos` 8D、`ik_abs` 7D、`ik_rel` 7D）、演示采集脚本与转换工具、以及解耦仿真/控制的同步与异步评测框架。
  - **未含于本仓：** 全仓检索 `reflexvla`、`dinov3`、`future_pred`、`qwen`、`vla_adapter`、`cuda_graph` 均无任何命中（仅 `README.md` 与 `CITATION.cff` 链接到项目页 `reflexvla.github.io`）。`scripts/evaluation/eval.py` 仅内置 `load_act_model` 本地测试路径与通用 `--backend server` 远端 HTTP 客户端；**ReflexVLA 的 1B 模型实现、多帧中间层因果 MHA 与 CUDA Graph 推理服务仍待论文接收后在官方项目页开放**。

## 源码架构核查

- **仿真任务注册（`source/reflexbench/`）：** 基于 Isaac Lab `ManagerBasedEnv` 注册 `*-Franka-{v0, Play-v0, IK-Abs-v0, IK-Rel-v0, DataCollection-v0}`，其中 `-DataCollection-v0` 开启固定俯视相机与腕部相机供 VLA 采数与评测。
- **数据采集（`scripts/data_collection/`）：**
  - `ConveyorBeltPickAndPlace` 因传送带动态抓取交互复杂，仅支持 `--policy checkpoint`（加载 `scripts/skrl/train.py` 训练的 PPO 检查点；`rsl_rl` 仅登记训练配置尚未接入采集器）。
  - 其余 5 个任务使用 `planning_policy.py` 中基于 cuRobo 与实时位置/速度前瞻预测的规划策略（如 `BallCatchingPlanningPolicy`、`RotatingPegInsertionPlanningPolicy` 等）。
  - `convert_to_lerobot.py` 读取 HDF5 元数据并导出为 LeRobot v3.0 数据集。
- **延迟感知评测框架（`scripts/evaluation/eval.py` + `policy_client.py`）：**
  - `Evaluator.run()` 初始化仿真与 `PolicyClient`，启动 RTF（Real-Time Factor, $t_{\mathrm{sim}}/t_{\mathrm{wall}}$）计时器；支持固定 `--infer_freq` 或 `--use_real_latency` 将实测壁钟 HTTP 请求耗时按 `_current_rtf()` 折算为仿真延迟步数。
  - **`--inference_mode sync`（`_run_sync`）：** 暂停等待 `/predict` 返回后，按折算延迟窗先执行 `_build_freeze_action()` 冻结机械臂（世界物理仍继续演化），再顺次执行最多 `--execution_horizon` 步动作。
  - **`--inference_mode async`（`_run_async` / `_run_async_realtime`）：** 通过 `PolicyClient.predict_async()` 在后台线程池提交 `/predict` 请求，同时环境以 `--ctrl_freq`（默认 25 Hz）继续执行上一个动作块（`active_actions`），首周期无可用动作时填补冻结动作，并在开启 `--async_time_align_actions` 时按已消耗时间跳过过期动作下标（`_elapsed_to_action_start_idx`）。
