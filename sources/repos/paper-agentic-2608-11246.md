# Thea 官方仓库核对归档（EIT-HAI/Thea）

- **仓库地址：** https://github.com/EIT-HAI/Thea
- **项目主页：** https://eit-hai.github.io/thea
- **关联论文：** [Towards the Harness of Embodied Agents 论文归档](../papers/paper-agentic-2608-11246.md)（arXiv:2608.11246）
- **默认分支：** `main`
- **许可证：** Apache-2.0（`LICENSE`）
- **开源状态：** **已开源**（公开提供提供方无关的 `thea-harness` 核心编排运行时、可选仿真接口 `thea-simulation`（LIBERO / RoboTwin 2.0）、Feishu/Lark 交互通道 `thea-lark` 及机器人移植模板 `examples/port-template/`；README 说明部署方需自行接入模型 API 凭据、具体机器人 SDK 或仿真器、底层动作策略、感知/场景图后端及后置条件证据）
- **核对边界：** 静态代码结构、README 与仓库目录样本核对，未本地运行

## 仓库目录与关键入口

基于 `EIT-HAI/Thea` `main` 分支 `README.md` 与目录文件样本核对：

1. **安装与多模式统一运行入口**
   - `./install.sh`：初始化 `.venv` 并安装 `thea-harness`、仿真接口、模型适配器与 Lark 通道包，同时生成基础配置文件。
   - `./run.sh`：统一调度四个运行模式（均支持 `--check` 预检）：
     - `--mode cli`（支持 `--instruction "..."` 单次运行，或进入交互式终端；亦可独立安装 `harness` 后直接运行 `thea-cli --config config.yaml`）
     - `--mode channel`（通过 Feishu/Lark 长连接 WebSocket 接入外部用户会话，暴露 `query_user` 与 `notify_user`）
     - `--mode simulation --harness-factory <module:create_harness>`（对接 LIBERO 或 RoboTwin 2.0 仿真并托管于 Lark 通道）
     - `--mode robot --harness-factory <module:create_harness>`（加载部署包的 `create_harness` 工厂函数连接实机 SDK）
   - Python 编程入口：`Harness.run_stream(instruction)`，流式输出模型决策、Tool Calls、Tool Results、评估结果与最终响应。

2. **核心编排运行时（`harness/`，独立包 `thea-harness`）**
   - `harness/cli.py`、`harness/protocols.py`、`harness/config.example.yaml`、`harness/pyproject.toml`
   - `harness/configuration/`：`loading.py`、`paths.py`、`runtime.py`（配置加载与运行时组装）
   - `harness/runtime/`：`core.py`、`agentic_loop.py`、`execution.py`、`execution_types.py`、`hooks.py`、`action_segments.py`、`standalone.py`、`task.py`（实现单轮单工具主循环、pre/post 钩子拦截、动作分段与任务生命周期）
   - `harness/context/`：`system_prompt.py`、`state.py`、`rendering.py`、`snapshots.py`、`compaction.py`、`redaction.py`（实现 Resident / Refreshed / Accumulated 三层上下文组装与历史压缩）
   - `harness/evaluation/`：`contract.py`、`model.py`、`runner.py`（独立于决策模型的后置条件评估与三态判决合同）
   - `harness/memory/`：`contracts.py`、`model.py`、`store.py`、`task_notes.py`（维护任务内 `Task Notes`、持久化 `MEMORY.md` 与 `tool_experience`）
   - `harness/skills/`：`discovery.py`、`runtime.py`（基于 `SKILL.md` 的三层渐进式技能发现与按需加载）
   - `harness/mcp/client.py`、`harness/tools/`、`harness/world/`（MCP 子进程工具客户端、工具注册表 `ToolRegistry`、Scene Graph 与 Embodiment Profile 接口、安全净空检查）
   - `harness/models/`：`client.py`、`providers.py`、`anthropic_adapter.py`、`openai_adapter.py`、`adapter_common.py`、`errors.py`（提供方无关的大模型适配器）
   - `harness/observability/run_log.py`（结构化运行日志记录）

3. **扩展与移植接口（`simulation/`、`lark/`、`examples/`、`docs/`）**
   - `simulation/README.md`：面向 LIBERO 与 RoboTwin 2.0 的 `thea-simulation` 适配器接口。
   - `lark/README.md`：面向飞书/Lark 机器人会话、授权与交互卡片的 `thea-lark` 通道。
   - `examples/port-template/`：`factory.py`、`profile.md`、`tools.py`、`README.md`（新机器人平台移植参考模板，含 Embodiment Profile 与工具注册样板）。
   - `docs/`：中英文集成与 API 文档站点源码（含 `docs/harness/port-to-your-robot.md`、`docs/scene-graph/index.md`、`docs/evaluation/index.md` 等）。

## 对 wiki 的映射

- [Towards the Harness of Embodied Agents (Thea) 实体页](../../wiki/entities/paper-agentic-2608-11246.md)
