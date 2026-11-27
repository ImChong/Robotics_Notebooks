# RoboClaw 官方仓库核对归档（RoboClaw-Robotics/RoboClaw）

- **仓库地址：** https://github.com/RoboClaw-Robotics/RoboClaw
- **关联论文：** [RoboClaw 论文来源归档](../papers/paper-agentic-2603-11558.md)（arXiv:2603.11558）
- **默认分支：** `main`
- **许可证：** GitHub 元数据未显式声明（仓库根目录文件样本中无独立 `LICENSE` 文件）
- **开源状态：** **部分开源**（Agent 编排框架、OpenClaw 风格 Skill 规范、MCP 服务示例与 TUI/GUI 交互入口已公开；README 说明当前主要在 Agibot G01 平台验证，完整包含 VLA 模型部署的 `.whl` 包需私信作者团队申请）
- **核对边界：** 静态代码结构与 README 核对，未本地运行

## 仓库目录与关键入口

基于仓库 `main` 分支根目录、README 与目录树样本核对：

1. **工程配置与启动入口**
   - `Makefile`、`pyproject.toml`、`uv.lock`、`.env.example`（配置 `OPENAI_API_KEY` 及可选 `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_VERIFICATION_TOKEN` / `FEISHU_EVENT_RECEIVER`）。
   - README 提供标准初始化与启动命令：`make init`、`make run_tui`、`make run_gui`。
   - 配置文件：`src/agent_demo/config/ormcp_services.json`、`src/agent_demo/config/project_openai_defaults.json`。

2. **Skill 规范库（`skills/`）**
   - `skills/eap-data-collection/SKILL.md`（配套 `references/data-collection-guardrails.md`、`references/eap-prompt-pairs.md`）：定义自重置 EAP 数据采集工作流与护栏。
   - `skills/long-horizon-execution/SKILL.md`（配套 `references/paper-vanity-four-tasks-spec.md`、`references/subtask-plan-template.md`）：定义长程任务拆解与梳妆台四子任务规格。
   - `skills/monitored-subtask-execution/SKILL.md`（配套 `references/mcp-tool-map.md`）：定义子任务受控执行与 MCP 工具映射。

3. **交互层与会话层（`src/agent_demo/interaction_layer/`、`session_layer/`）**
   - TUI / GUI / CLI / 飞书入口：`tui/olympus_tui.py`、`gradio_ui/gradio_ui.py`、`cmd/olympus_cmd.py`、`cmd/olympus_img_cmd.py`、`feishu_bot.py`、`feishu_long_connection.py`、`local_skill_support.py`。
   - 会话与工作流：`session_layer/session_manager.py`、`base_session/base_session.py`、`intention_classification/intention_classification.py`、`workflow/base_workflow/base_workflow.py`。

4. **智能体核心与组件层（`src/agent_demo/agent_layer/`）**
   - 核心智能体：`agent_core/planning_agent/planning_agent.py`、`react_agent/react_agent.py`、`act_agent/act_agent.py`、`img_act_agent/img_act_agent.py`、`base_agent/base_agent.py`。
   - 记忆、LLM 与工具组件：`memory_manager/memory_manager.py`、`runtime_memory_tree/runtime_memory.py`、`llm_manager/openai_client/openai_client.py`、`agent_tools/local_skill_registry.py`、`agent_tools/agent_tools.py`、`ormcp_service_manager/ormcp_service_manager.py`、`ormcp_service_manager/ormcp_service_connection.py`。
   - 提示词与数据加载：`agent_prompt/references/RoboClaw.md`、`agent_prompt/references_loader.py`、`machine_layer/dataloader_a2d.py`、`machine_layer/dataloader_corobot.py`。

5. **MCP 服务端示例（`src/mcp_server_demo/`）**
   - `corobot_mcp_server/src/server.py`（机器人控制 MCP 服务）
   - `data_analyst_mcp_server/src/server.py`（数据分析 MCP 服务，含 `test_analyzer.py`）
   - `gma_mcp_server/src/server.py`（GMA MCP 服务）

## 对 wiki 的映射

- [RoboClaw 实体页](../../wiki/entities/paper-agentic-2603-11558.md)
