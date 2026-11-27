# project-m2hri/m2hri — M2HRI 官方开源仓库档案

- **GitHub 地址：** https://github.com/project-m2hri/m2hri
- **默认分支：** `main`
- **开源协议：** 未在仓库根目录或 GitHub 元数据中声明独立 `LICENSE` 文件
- **关联论文：** *M2HRI: An LLM-Driven Multimodal Multi-Agent Framework for Personalized Human-Robot Interaction*（[arXiv:2604.11975](https://arxiv.org/abs/2604.11975)，[项目页](https://project-m2hri.github.io/)）
- **核对边界：** 静态代码核对，未本地运行（已核查 `README.md`、`files_sample`、`ma_server.py`、`ma_clients.py` 及 `modules/`、`config/` 核心源码文件，未在本地安装 Python 2.7 NaoQi SDK 或连接实体 NAO 机器人运行）

## 核心入口与目录结构（已核实）

- **双进程主入口：**
  - `ma_server.py`（Python 3.10+）：LLM 服务端入口，加载 `config/agent_config.json` 与 `config/prompts.json`，为每台机器人构造 `NaoAgent`（继承 AutoGen `AssistantAgent`），通过 `SelectorGroupChat`、`UserProxyAgent` 与 `LLMTurnManager` 驱动多轮多智能体交互，并经 ZMQ REQ 端口向客户端下发工具调用；支持 `--agent`、`--config`、`--listen`（麦克风语音识别）、`--no-memory`、`--zmq-port`、`--zmq-host`、`--robot-ip`。
  - `ma_clients.py`（Python 2.7）：NAO 机器人客户端，绑定 ZMQ REP 端口（支持 `--multi` 单端口多机分发或 `--slot N` 单机模式），接收 `{agent, tool, args}` JSON 指令；支持 `--connection text`（仅终端打印，免实体机器人）或 `--connection speech --mode execute`（通过 NaoQi `ALProxy("ALVideoDevice", ...)` 持续抓取 `see.jpg`、开启人脸追踪并调用 `modules/actions/actions.py` 执行动作）。
- **核心认知与协调模块（`modules/`）：**
  - `modules/llm/turn_manager.py`：`LLMTurnManager` 实现 `vlm_selector` 与 `candidate_func`，在人类发言后并发调用视觉感知与记忆检索，同时用 `gpt-4o`（`_vlm_score_response_sync`，`VLM_THRESHOLD = 0.5`、`VLM_TOP_K = 2`）对各机器人打分并按降序排队发言，机器人发言后再触发记忆写入。
  - `modules/llm/perception_vlm.py`：`perceive_image()` 读取 `session/<robot_name>/see.jpg` 并编码为 base64 调用 `gpt-4o-mini` 生成场景描述，写入 `perception.txt`。
  - `modules/llm/helpers.py`：根据 `modules/actions/tools.json` 动态生成带参数签名的异步工具函数（`_make_nao_tool_with_signature`）并管理 ZMQ REQ 发送（`_create_nao_socket`、`_send_nao_command_sync`）。
  - `modules/llm/SessionLogger.py` & `modules/llm/input_names.py`：管理 `session/<robot_name>/` 目录及 `.jsonl` 会话日志。
  - `modules/memory/memory_agent.py` & `modules/memory/memory.py`：基于 LangGraph `StateGraph` 构建 `MemoryAgent`，操作 ChromaDB 的 `semantic_memory`、`episodic_memory` 与 `procedural_memory` 三个集合（`text-embedding-3-large` 嵌入）；`--no-memory` 时切换为 `NoMemoryAgent` 与 `BufferedChatCompletionContext(buffer_size=1)`。
  - `modules/personality/personality_module.py` & `modules/personality/big5.json`：`PersonalityEngine` 读取或初始化 `session/<robot_name>/personality.json`，将 1–5 分 Big Five（OCEAN）数值转为系统提示文本，并提供 `is_antisocial()` 检查。
  - `modules/actions/actions.py` & `modules/actions/tools.json`：定义 15 个 NAO 动作工具原语（`speak`、`rest`、`wave`、`sit`、`stand`、`nod`、`shake_hand`、`walk`、`adjust_volume`、`track_face`、`point_at`、`rest_arm`、`rest_body`、`move_head`、`wait`）。
- **配置与环境初始化：**
  - `config/agent_config.json`、`config/nao_config.py`（`NAO_BASE_PORT = 5555`、`ROBOT_IPS`）、`config/prompts.json`（含 `nao_agent_system`、`perception_vlm_system`、`vlm_score` 三套提示词模板）。
  - `initialize/llm_requirements.txt`、`initialize/nao_requirements.txt`、`initialize/set_llm_path.sh`、`initialize/set_nao_path.sh`。

## 复现与运行依赖边界

- **双环境隔离要求：** 服务端需 Python 3.10+ 与有效 `OPENAI_API_KEY`（`gpt-4o-mini`、`gpt-4o`、`text-embedding-3-large`）；实体机器人在客户端侧强依赖 Python 2.7 与 SoftBank NaoQi SDK，二者通过 ZMQ TCP 通信解耦。
- **无硬件验证路径：** 可在本地或单机启动 `python ma_clients.py --multi --connection text` 与 `python ma_server.py --no-memory`（或开启 ChromaDB 记忆）验证文本对话、多智能体轮次调度与工具指令序列化，但缺少真实摄像头帧 `see.jpg` 时 `perceive_image` 与 `_vlm_score_response_sync` 将回退为空感知与默认分 `0.5`。
- **代码与论文差异记录：** 论文 §III-A2 与 §III-C 正文主要叙述工作记忆、语义记忆与情景记忆（ChromaDB 两类长期集合），而仓库 `modules/memory/memory.py` 额外包含 `procedural_memory` 集合与对应检索/存储工具，且 `turn_manager.py` 硬编码了 `VLM_THRESHOLD = 0.5` 与 `VLM_TOP_K = 2`。

## 与 wiki 的映射

- 对应实体页：[M2HRI: An LLM-Driven Multimodal Multi-Agent Framework for Personalized Human-Robot Interaction](../../wiki/entities/paper-agentic-2604-11975.md)
