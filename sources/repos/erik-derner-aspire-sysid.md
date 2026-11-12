# ASPIRE SysID — 官方源码仓库（erik-derner/aspire-sysid）

> 来源归档（repo）；对应 [ASPIRE 论文条目](../../wiki/entities/paper-aspire.md)。注意：此项目与 NVIDIA GEAR 的同名 ASPIRE 机器人技能系统不是同一个项目。

- **项目：** ASPIRE: An Agentic System-Identification Platform for Iterative Reasoning and Exploration
- **官方代码：** <https://github.com/erik-derner/aspire-sysid>
- **论文：** <https://arxiv.org/abs/2611.13114v1>（2026-11-09）
- **类型：** repo（LLM 智能体 + 系统辨识 harness）
- **开源状态：** 已公开 Python harness、任务数据、实验运行记录、PySINDy / PySR 对照材料；仓库标示 CC BY 4.0
- **核对日期：** 2026-11-12
- **核对范围：** 静态阅读官方根目录 README、`aspire/README.md`、`aspire/pyproject.toml`，以及 `cli.py`、`experiment/commands.py`、`experiment/worker.py`、`experiment/runner.py`、`agent/loop.py`、`brief/workspace.py`、`task/provision.py`、`evaluate/pool.py` 中相关入口与调用；未执行软件、未逐一审阅全部运行记录。

## 运行入口（以官方代码为准）

- `aspire` 控制台脚本来自 `aspire/pyproject.toml`：`aspire = "aspire_sysid.cli:main"`。
- `aspire experiment run experiments/paper-turtlebot --dry-run`：`cli.py` 注册 experiment 子命令，`experiment/commands.py` 的 `_cmd_run` 调用 `grid.run_grid`；网格通过 `worker.py` 调用 `runner.execute_run`。
- `runner.execute_run` 读取任务和预算、创建运行目录，调用 `brief.workspace.provision_workspace` 写入 agent 可见的 `TASK.md` / `FORMAT.md` / 训练与验证数据 / 检查脚本；通过 `agent.loop.run_round` 使用模型和工具循环生成候选池。
- `agent.loop._run` 在收到严格的 `DONE` 完成消息后调用 `model.check_workspace`；通过检查后由 `runner` 提供的回调 `evaluate_pool(task, report)` 评估；若首次检查失败，允许一次修正机会（预算内）。
- `evaluate.pool.evaluate_pool` 对候选做表达式检查、形式化要求核验、轨迹误差计算、资格与测试集排名；模型推荐由智能体在看到测试误差前写入 `models/pool.json`，测试集优胜者的 oracle 排名另记。
- `runner.execute_run` 随后保存候选快照、工作空间副本，运行 `leak_scan`，写入运行记录并导出 Python / Matlab 结果；`aspire record verify` 可在本地离线重算评价。

## 信任边界与限制（官方 README 明确说明）

- `task/provision.py` 只复制 train / val 数据和配套文件到工作空间；测试数据归 harness 的 sealed evaluation 模块读取，正常接口不会提供给智能体。
- **这不是 OS 级安全沙箱。** 官方 `aspire/README.md` 的 “What the harness does not guarantee” 指出，智能体命令和 harness 使用同一操作系统用户，且 shell 可访问网络；智能体程序理论上可能读取可见的任意文件，包括测试集。命令过滤与目录分离**不保证**阻断恶意访问；事后 leak scan 也无法发现所有静默读取。
- 因此“封存测试集”应理解为 **harness 的正常工具/工作目录及消息访问边界**，不是对抗性执行环境的安全隔离。需要严格隔离时，应自行采用容器、VM 或不同 OS 用户（官方文档建议）。
- 这些是静态源码与文档检查结论；本次未进行端到端运行或渗透测试。

## 来源定位

- [根 README](https://github.com/erik-derner/aspire-sysid/blob/main/README.md)
- [harness README（含信任区）](https://github.com/erik-derner/aspire-sysid/blob/main/aspire/README.md)
- [CLI 注册](https://github.com/erik-derner/aspire-sysid/blob/main/aspire/aspire_sysid/cli.py)
- [实验子命令](https://github.com/erik-derner/aspire-sysid/blob/main/aspire/aspire_sysid/experiment/commands.py)
- [工作进程](https://github.com/erik-derner/aspire-sysid/blob/main/aspire/aspire_sysid/experiment/worker.py)
- [单次运行](https://github.com/erik-derner/aspire-sysid/blob/main/aspire/aspire_sysid/experiment/runner.py)
- [智能体回合](https://github.com/erik-derner/aspire-sysid/blob/main/aspire/aspire_sysid/agent/loop.py)
- [候选评估](https://github.com/erik-derner/aspire-sysid/blob/main/aspire/aspire_sysid/evaluate/pool.py)
- [ASPIRE 论文归档](../papers/aspire_arxiv_2611_13114.md)
