# OpenMOSS/OpenETA 源码与仓库核对记录

- **官方仓库：** [https://github.com/OpenMOSS/OpenETA](https://github.com/OpenMOSS/OpenETA)
- **项目主页：** [https://openmoss.ai/OpenETA/](https://openmoss.ai/OpenETA/)
- **开源协议：** Apache-2.0（`LICENSE`）
- **默认分支：** `main`（另有专题开发分支 `openeta-light`、`dev/real-robot-deployment`、`robodojo-v112` 等）
- **开源状态：** 已开源（包含完整可执行框架代码、CLI/MCP 入口、仿真与真机适配层及评测脚本）
- **核对边界：** 静态代码核对（基于仓库文件目录抽样 `files_sample`、仓库页面快照与 `README.md` 预览，未在本地拉起环境运行）

## 目录结构与关键入口（已核实）

- **顶层工程配置与文档：** `pyproject.toml`、`uv.lock`、`.env.example`、`.mcp.example.json`、`CONTEXT.md`、`README.md`、`README.zh-CN.md`、`η0_Embodied Task Agent.md`。
- **核心 Agent 运行时（`agent/runtime/`）：**
  - `runtime.py`、`runtime_assembly.py`、`episode.py`、`rollout.py`：主闭环调度、回合生命周期与不可变轨迹（`rollout/`）归档。
  - `planner.py`、`planner_prompts.py`、`conversation.py`：Planner 交互与受限多模态对话上下文管理。
  - `interfaces.py`、`actions.py`、`checkers.py`、`supervision.py`、`sandbox.py`：Host 侧 Tool 契约校验、side-effect 分类门控（`read_only`/`planning`/`bookkeeping`/`world_mutating`）、安全检查与原子动作分发。
  - `memory.py`、`memory_store.py`、`promoted_memory.py`、`skills.py`、`skill_authoring.py`、`self_improvement.py`、`task_playbooks.py`：工作记忆、Skill 文本指导与受约束自我演化晋升逻辑。
  - `depth_contract.py`、`depth_enhancement.py`、`calibration.py`、`calibration_registry.py`、`reference_localization.py`：深度契约与增强、标定生命周期及目标参考定位。
- **Planner 后端与 GUI（`agent/backends/`、`agent/gui/`）：**
  - `planner.py`、`policy_api.py`、`code_policy.py`、`provider_config.py`：OpenAI 兼容 LLM/VLM 接口（支持主备端点切换）、策略 API 与受限代码策略后端。
  - `agent/gui/provider_config_app.py`：提供方配置界面。
- **CLI 与评测套件（`agent/cli/`、`agent/evals/`）：**
  - `agent/cli/openeta_cli.py`（对应 `uv run openeta` 交互 TUI 与 `--once` / `--command preflight|run|iterate|inspect` 入口）。
  - `agent/cli/batch_eval.py`（对应 `uv run openeta-batch` 并行 manifest 评测入口）。
  - `agent/cli/eval.py`、`agent/cli/experiment.py`、`agent/cli/visual_state_eval.py`、`agent/cli/visual_history_eval.py`、`agent/cli/subagent_eval.py`。
  - `agent/evals/runner.py`、`tool_contract_audit.py`、`tool_contract_conformance.py`、`tool_contract_promotion.py` 等工具契约审计与晋升测试模块。
- **适配器与轻量 Codex 插件（`adapter/`）：**
  - `adapter/openeta_agent.py`、`adapter/codex_cli.py`、`adapter/bridge.py`、`adapter/protocol.py`、`adapter/sim.py`、`adapter/motion_profiles.py`：连接 Codex CLI/TUI 与仿真器/真机后端的轻量协议桥接。
- **仿真与真机后端（`sim/`、`real/`、`tools/`、`evaluations/`、`examples/`）：**
  - `sim/mcp_server`（`uv run python -m sim.mcp_server`）：仿真环境 MCP 服务、REST API 与 Web Dashboard，支持 MetaWorld、ManiSkill、LIBERO、RoboCasa365、BEHAVIOR-1K 与 Genesis（初始仿真适配借鉴并迁移自 RLinf）。
  - `real/`：真机部署栈（支持 RealSense D400/L515、Webcam/RTSP 与 UR5e 机械臂状态/运动接口，Franka 在矩阵中尚未标记适配）。
  - `agent/skills/`（`pick.md`, `place.md`, `push.md`, `pull.md`, `stack.md`, `embodiment_explore.md`, `memory_extract.md`, `sim_mcp.md`）与 `agent/strategies/grasp/candidate/`：可编辑技能文本与抓取策略 JSON 配置。

## 对应知识页

- [ETA: A New Agentic Paradigm for Embodied Tasks](../../wiki/entities/paper-agentic-2608-03924.md)
