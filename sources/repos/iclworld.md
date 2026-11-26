# ICLWorld：公开实现与运行边界

- **源码：** [ICLWorld/ICLWorld](https://github.com/ICLWorld/ICLWorld)，本次静态核查提交 [2ce5316](https://github.com/ICLWorld/ICLWorld/tree/2ce5316c33da9b2eeb2cb22164e40ab02ea6b044)。
- **许可：** 主仓代码 MIT，原生 CLI、第三方资源和数据媒体不随主仓 MIT 授权；参见[许可清单](https://github.com/ICLWorld/ICLWorld/blob/2ce5316c33da9b2eeb2cb22164e40ab02ea6b044/docs/licenses.md)。
- **核对范围：** 阅读 README、robot/README、数据合同及 Web/OS 计划生成、单次执行和分数归档链路；独立核对机器人入口，不执行模型或环境。
- **通读：** 已完成

- **Web/OS：** 官方 `scripts/prepare_dataset.py` 整理样例，`scripts/init_env.py` 与 Docker Compose 提供交互环境；`scripts/experiment.py` 可预览或启动，调用 `runner/run_one.py`，后者带入各条件的可公开输入并调用 `native.py`。评测方读取权威终态结果，检查任务条件和哈希后写入回执，`scripts/scores.py` 汇总。默认 Harbor 工作区用于组织任务，不是直接用 Harbor 默认为智能体提供 shell。
- **Robot：** 独立的 `robot/scripts/prepare.py` 检查并组装任务和资源，`robot/harness/pool_verified.py` 用原生仿真与浏览器 GUI 启动隔离实例。依赖匹配的 NVIDIA GPU/驱动及 IsaacSim/IsaacLab。官方 `--probe-only` 不调用模型，可分离检查环境和 GUI；示例中的工程运行开关不表示 GPU 跑分已验证。独立 `robot/qa/evaluate.py` + `grade.py` 不访问仿真状态或动作。
- **边界：** 模型/CLI/API 访问由运行者提供；公开 Goal、Video/Text 规则、裁判状态与答案键分离。基础设施异常保留未计分，终止的模型失败才是有效失败。入口可审查不代表在线 API 能复现同样输出。

- **Wiki：** [ICLWorld](../../wiki/entities/paper-iclworld.md)
