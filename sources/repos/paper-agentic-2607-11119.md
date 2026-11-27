# hengyuan-hu/via 官方代码核查

- **代码仓库：** <https://github.com/hengyuan-hu/via>
- **开源协议与分支：** MIT License，默认分支 `main`
- **对应论文：** [VIA: Visual Interface Agent for Robot Control（arXiv:2607.11119v2）](../papers/paper-agentic-2607-11119.md)
- **核对范围：** 静态核对 GitHub 仓库主页快照、README 安装与使用说明、根目录及 `via/`、`prompts/` 文件清单（含 `pyproject.toml`、`uv.lock`、`.mcp.json`、`set_env.sh`、`via/mcp_server.py`、`via/run_eval.py`、`via/record_sim.py`、`via/sim_env.py`、`via/rainbow_env.py`、`via/t_block_env.py`、`via/harness.py`、`via/claude_harness.py`、`via/codex_harness.py`、`via/UI/template_index.html` 等条目）。静态代码核对，未本地运行。
- **通读：** 未完成
- **开源范围判断：** 已开源仿真环境、Browser 3D 交互界面、Playwright 驱动的 MCP Server 及 Claude Code / Codex 批量评测脚本；LIBERO 通过 `third_party/LIBERO` git submodule 引入。仓库当前结构与 README 未列真机 Stretch 4 控制端或相关硬件驱动代码。

## 已核对目录与入口

1. **环境与依赖：** `pyproject.toml` 与 `uv.lock` 管理 Python 3.10 环境（含 `bddl`、`gym`、`future`、`easydict`、`cloudpickle`、`torch`/`torchvision` 及 `playwright` Chromium）；`set_env.sh` 激活 `.venv` 并将仓库根目录与 `third_party/LIBERO` 加入 `PYTHONPATH`。
2. **核心包 `via/`：**
   - 仿真与任务环境：`via/sim_env.py`、`via/rainbow_env.py`、`via/t_block_env.py`、`via/robot_utils.py`、`via/episode_recorder.py`。
   - 交互界面与服务：`via/UI/template_index.html` 与 `franka.obj` / `franka_closed.obj` / `robotiq.obj` / `robotiq_closed.obj` 夹爪三维模型、`via/serve.py`、`via/mcp_server.py`（由 `.mcp.json` 自动拉起）。
   - 智能体评测与记录：`via/record_sim.py`（单任务交互运行，如 `python -m via.record_sim --task stack`）、`via/run_eval.py`（多种子并行评测，如 `python -m via.run_eval stack-opus-dev -j3`）、`via/harness.py`、`via/claude_harness.py`、`via/codex_harness.py`、`via/agent_common.py`。
3. **任务提示词 `prompts/`：** 包含 `bowl_plate*.md`、`open_drawer*.md`、`rainbow_min.md`、`t_block_min.md`、`turn_stove*.md`。

## 映射

- [VIA 论文实体与源码运行时序图](../../wiki/entities/paper-agentic-2607-11119.md)（静态核对 README 与文件入口，未本地运行仿真或连接 Claude/Codex API）。
