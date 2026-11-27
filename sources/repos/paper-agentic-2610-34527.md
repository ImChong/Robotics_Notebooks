# LLM4Robot-Bench（原 PhysLadder）官方开源仓库

- **仓库地址：** <https://github.com/LLM4Robot/LLM4Robot-Bench>
- **历史归档地址：** <https://github.com/PhysLadder-Benchmark/PhysLadder>（2026-11-08 归档，README 指引全部新实验与更新使用 `LLM4Robot/LLM4Robot-Bench`）
- **项目网站：** <https://llm4robot.github.io/bench/>
- **对应论文：** [arXiv:2610.34527](https://arxiv.org/abs/2610.34527)（[论文原文核对笔记](../papers/paper-agentic-2610-34527.md)）
- **默认分支：** `main`（历史归档仓为 `master`）
- **开源协议：** Apache-2.0（团队自研代码；第三方资产遵循 `NOTICE` 与 `docs/release/asset_permissions.json`）
- **核对方式与边界：** 静态代码核对（基于 GitHub 仓库 `README.md` 与目录/文件采样 `files_sample`），未本地运行；仿真执行需 NVIDIA Isaac Sim 5.1、兼容 NVIDIA GPU、独立 pinned cuRobo 环境、Git submodule 仿真资产及模型 API 访问权限。

## 仓库结构与已核实入口

根据 `LLM4Robot/LLM4Robot-Bench` 仓库根目录与 README 文档，项目核心模块与入口如下：

- **核心框架代码（`src/robot_tasks/`）：** 共享任务执行、策略适配器、运动规划（cuRobo 接口）、任务定义与自动评测器。
- **命令行工具（`tools/`）：**
  - `tools/verify_runtime_assets.py`：校验通过 `git submodule update --init assets` 拉取的 USD 仿真资产与 `runtime.lock.json` SHA-256 哈希。
  - `tools/release/check_task_contract.py`：CPU 侧校验任务契约配置。
  - `tools/launch.py`：列出并管理基准任务设置（`python tools/launch.py list`）。
  - `tools/reproduce.py`：打印执行计划（不带 `--execute` 时做离线 dry run）或实际启动仿真评测（`--execute`，支持 `--tasks`、`--suite original|variant01|all`）。
  - `tools/batch_run.py`：批量汇总运行输出目录（`--summarize runs/PATH_TO_BATCH`），区分任务失败、超时、中断与基础设施错误。
  - `tools/experiments/bottle_preloosen.py`：额外 L3-01 270° 预松动配置检查与启动脚本（配合 `configs/experiments/l3_01_preloosen_270.json`）。
- **配置与任务定义（`configs/`）：**
  - `configs/benchmark/`：`experiments.json`、`task_contract.json`、`task_numbering.json`。
  - `configs/tasks/`：16 项任务的原始设置与 `variant01` 变体，共 32 组可运行配置（默认预算 150 次模型调用或 5 小时）。
  - `configs/models/gpt.toml`：模型接入配置；`configs/robots/r5.toml`、`r5_sim.toml` 等：ARX R5 机械臂配置；`configs/cameras/`：外视角与腕部相机标定配置；`configs/runtime/`：`local.example.toml`、`codex.example.toml`、`planner.example.toml`。
- **资产与测试（`assets/`、`skills/`、`tests/`）：**
  - `assets/`：独立子模块 `PhysLadder-Assets`，含 `runtime.lock.json` 与 USD 场景/物体资产。
  - `skills/r5-remote-deploy/SKILL.md`：可选远程部署技能说明。
  - `tests/`：单元测试集（`python -m unittest discover -s tests -v`）。
- **复现边界：** 仓库主要提供仿真评测框架、资产及文档化的评测协议；真机控制与执行记录在文档中与仿真运行时分开说明，不包含真机闭源 SDK 的现场自动化标定。
