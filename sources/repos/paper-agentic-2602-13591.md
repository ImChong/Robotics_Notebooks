# PKU-LLM-DS-LAB/AgentRob 官方代码仓库核对

- **仓库地址：** <https://github.com/PKU-LLM-DS-LAB/AgentRob>
- **关联论文：** [arXiv:2602.13591](https://arxiv.org/abs/2602.13591)（`AgentRob: From Virtual Forum Agents to Hijacked Physical Robots`）
- **默认分支：** `main`
- **开源状态：** 部分开源（已包含 Agent 层 Python 轮询代理与 Robot 层 VLM 控制驱动脚本；TypeScript MCP Server 与 NodeBB 论坛部署说明为单独维护，未包含在本仓库内）
- **许可证：** 仓库未检出独立 LICENSE 文件；`README.md` 声明由北京大学研究者开发，授权事宜需联系作者
- **核对边界：** 静态代码核对，未本地运行（基于仓库目录采样与 `README.md` 完整内容核对，未安装 `unitree_sdk2py`、未连接真实 Unitree Go2/G1 硬件或外部 Volcengine ARK/TOS 与 NodeBB 服务）

## 已核对目录与入口文件

仓库根目录共包含 10 个核心文件/目录条目（由 GitHub 文件列表采样确认）：

| 文件/目录 | 角色与已核对说明 |
|---|---|
| `go2_mcp_agent.py` | Unitree Go2 论坛代理入口（`Go2ForumAgent`，`AGENT_ID=go2-mcp-agent`），监听 `@quadruped` / `@机器狗`；通过 stdio 子进程或 WebSocket（`--mcp-mode network`）连接 MCP Server，轮询论坛帖子、提取指令并调用驱动，最后回帖汇报 |
| `g1_mcp_agent.py` | Unitree G1 论坛代理入口（`G1ForumAgent`，`AGENT_ID=g1-mcp-agent`），监听 `@humanoid` / `@g1` / `@机器人`，流程与 Go2 代理对称 |
| `robot_command_driver.py` | 统一机器人驱动调度模块，提供 `drive_go2_robot()` 与 `drive_g1_robot()` 异步接口，负责初始化网络接口（默认 `eth0`）上的 Unitree DDS 通信并调度对应 VLM 控制器 |
| `go2_VLM_client.py` | Go2 四足机器人 VLM 控制器与交互式 CLI 客户端（`Go2VLMController`），暴露 `act_move`、`act_hello`、`act_heart`、`act_backflip` 与 `get_front_image`、`img_to_volc` 原语供 VLM 迭代 tool calling |
| `g1_VLM_client.py` | G1 人形机器人 VLM 控制器与交互式 CLI 客户端（`G1VLMController`），暴露 `act_move`、`act_hello`（手臂动作客户端挥手）及 USB 摄像头感知/上传原语 |
| `README.md` / `README_robot.md` | 系统架构文档、环境变量配置（`.env`）、CLI 参数说明、多模式启动示例及机器人联调排错指南 |
| `assets/overview.png` / `assets/architecture.png` | 论文 Figure 1 与三层架构示意资源 |
| `.gitignore` | 忽略 `.env` 等本地敏感密钥文件 |

## 外部运行依赖核查

- **Python 依赖**：`tqdm`、`opencv-python`、`websockets`、`python-dotenv`、`tos`、`volcenginesdkarkruntime`，以及对应真机环境的 `unitree_sdk2py`（要求 Linux 与 Python 3.9+）。
- **外部服务与单独维护组件**：
  - **NodeBB 论坛实例**：作为 Forum Layer 持久化通信平台（必需）。
  - **TypeScript MCP Server**：基于 `@modelcontextprotocol/sdk` 与 `zod` 实现 8 个 JSON-RPC 2.0 论坛工具，通过 `NODEBB_MCP_ROOT` / `NODEBB_MCP_ENTRY`（默认 `src/index.ts`，依赖 Node.js 18+ `npx tsx`）或远程 WebSocket（`ws://<HOST>:8765/mcp`）接入，**单独维护，不在本仓库内**。
  - **Volcengine ARK（默认模型 `doubao-seed-1-8-251228`）**：承担指令提取、结果总结与 VLM 工具调用（支持切换 OpenAI 兼容接口或本地模型）。
  - **Volcengine TOS**：用于 `img_to_volc()` 上传机器人采集画面并返回图片 URL（可选）。

## 对 wiki 的映射

- [AgentRob: From Virtual Forum Agents to Hijacked Physical Robots](../../wiki/entities/paper-agentic-2602-13591.md)
