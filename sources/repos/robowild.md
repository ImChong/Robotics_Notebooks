# RoboWild 仿真基准与 RoboWild-Harness 运行代码

- **类型：** repo
- **官方 RoboWild：** https://github.com/xlang-ai/RoboWild
- **官方 Harness：** https://github.com/xlang-ai/RoboWild-Harness
- **项目：** https://robowild.xlang.ai/
- **论文：** https://huggingface.co/datasets/xlangai/RoboWild/blob/main/RoboWild.pdf
- **数据：** https://huggingface.co/datasets/xlangai/RoboWild
- **许可：** RoboWild 仓库根 LICENSE / README 标示 Apache-2.0；第三方仿真资产及依赖需各自核查
- **核对日期：** 2026-11-22
- **核对范围：** RoboWild README（规模、环境安装、ReAct 设置、评测命令与结果文件），docs/agent_evaluation.md（跨本体与 Coding agent），docs/task_definitions.md（config.yaml/evaluator.py），docs/runtime.md（控制模式），docs/development.md（结构），test.py（仿真烟测）；source/robowild/robowild/bench/env/isaaclab_env.py 中 IsaacLabEnvWrapper 的 reset/step/evaluate；RoboWild-Harness README、docs/TOOL_PROFILES.md 及 src/robowild_harness/cli.py 评测 CLI 入口。未下载完整资产或实际启动 Isaac Sim。
- **通读：** 已完成

## 运行与复现边界

- **任务组成：** 239 个 Dexmate Vega-1 原始设置，34 个设置分别复用到 Mobile ALOHA、Franka Mobile、Unitree G1 + Dex3，共 239 + 34×3 = 341 个任务—本体组合；不是 341 个不重复的场景任务。
- **平台依赖：** Python 3.12、Isaac Sim 6.0.1、Isaac Lab 3.0.0-beta2、NVIDIA GPU、CUDA 12.8 工具链、数据资产和 cuRobo；README 提供 scripts/install_simulator.sh 与手动安装两种方案，安装体积和驱动版本需按官方核查。
- **默认协议：** `backend: react`，`--tool-profile react_move`，`--mode independent`，RGB 相机观察；仿真在代理推理间暂停。Short 任务官方建议 `call_limit=100`，Long 任务为 200；`done()` 不计调用。
- **模型接口：** ReAct 用 Responses-compatible API 与模型交互，密钥写入环境变量，不放入 JSON；`react_move` 仅提供 `move_robot` + `done`，动作工具内部执行规划+运动，默认规划不使用环境障碍碰撞世界，需避免把成功率混同于安全性能。
- **Coding agent 单独路径：** `--tool-profile full`，生成 robot.py 接口，通过工具显式查状态、规划、执行；与 ReAct 榜单不应混为同一协议。
- **任务规则：** `source/robowild/robowild/tasks/<group>/<task-id>/config.yaml` 定义指令/场景/本体设置；旁边 `evaluator.py` 执行可信成功判据。Harness 负责 ReAct、Robot API、规划、会话、预算与日志；RoboWild 负责仿真状态与 evaluator。
- **执行结果：** Harness 按 README 保存 `manifest.json`、`batch.json`、`report.json`、`summary.md`；单轮评测与多轮统计须单独注明。

## 官方快速开始（前提是先完成安装与资产下载）

```bash
# 在 RoboWild workspace 根目录完成 setup 后
source activate_robowild.sh
CUDA_VISIBLE_DEVICES=0 ./IsaacLab/isaaclab.sh -p robowild/test.py --enable_cameras

cd robowild
# 将模型配置写入 react.json，并设置 AGENT_CONFIG / MODEL_API_KEY
options=$(python -c 'import json,os; print(json.dumps(dict(robot_type="Dexmate",control_mode="joint_position",assets_dir=os.environ["ASSETS_DIR"],curobo_source=os.environ["CUROBO_SOURCE"])))')
robowild-harness eval --environment robowild --environment-checkout "$PWD" \
  --tasks 01-01 --agent "$AGENT_CONFIG" --assignment-options "$options" \
  --tool-profile react_move --mode independent --gpus 1 --envs-per-gpu 1 --rounds 1
```

## 已核对代码指向

- `RoboWild/source/robowild/robowild/bench/env/isaaclab_env.py`：`IsaacLabEnvWrapper.reset`、`step`、`evaluate`。
- `RoboWild/source/robowild/robowild/tasks/`：场景任务、配置与可信 evaluator；`RoboWild/test.py`：`load_task → prepare_task → make_env_cfg → make_env → reset/step/evaluate` 烟测。
- `RoboWild-Harness/src/robowild_harness/cli.py`：`robowild-harness eval` 命令参数与调度入口。
- `RoboWild-Harness/docs/TOOL_PROFILES.md`：`react_move` 的输入/输出和环境碰撞限制。
- `RoboWild-Harness/README.md`：ReAct → Robot API → Gateway → Ray sessions → Isaac Lab env 的模块所有权界面。

**未执行：** 本次只做官方文档及上述源码静态核对，未声称真实复现或复测原榜单。

## Wiki 映射

- [RoboWild 论文详情](../../wiki/entities/paper-robowild.md)
- [官方项目页归档](../sites/robowild-xlang-ai.md)
