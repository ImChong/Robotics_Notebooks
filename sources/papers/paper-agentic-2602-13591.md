# AgentRob: From Virtual Forum Agents to Hijacked Physical Robots

- **作者：** Wenrui Liu, Yaxuan Wang, Xun Zhang, Yanshu Wang, Jiashen Wei, Yifan Xiang, Yuhang Wang, Mingshen Ye, Elsie Dai, Zhiqi Liu, Yingjie Xu, Xinyang Chen, Hengzhe Sun, Jiyu Shen, Jingjing He, Tong Yang
- **机构：** 北京大学（Peking University）
- **版本 / 载体：** arXiv:2602.13591v1 [cs.RO]（2026-02-14 提交，10 页，2 图，2 表；CC BY 4.0）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2602.13591)、[arXiv 摘要页](https://arxiv.org/abs/2602.13591)
- **项目页 / 代码：** 部分开源（[PKU-LLM-DS-LAB/AgentRob](https://github.com/PKU-LLM-DS-LAB/AgentRob)；Agent 与机器人控制层 Python 代码已开源，TypeScript MCP Server 与 NodeBB 论坛部署单独维护）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、Sections 1–7、Table 1–2、Figure 1、Algorithm 1、Listing 1 及 References），并核对官方 GitHub 仓库首页、`README.md` 预览及文件树清单（`go2_mcp_agent.py`、`g1_mcp_agent.py`、`robot_command_driver.py`、`go2_VLM_client.py`、`g1_VLM_client.py`、`README_robot.md`）
- **通读：** 已完成

## 原文核对笔记

- **核心问题与动机**：现有 LLM 自主智能体（如 AutoGPT、BabyAGI、MetaGPT）主要局限于虚拟环境，而自然语言控制机器人工作（如 SayCan、Code as Policies、RT-2）通常依赖专用 API、实时点对点连接与专业部署环境。AgentRob 提出将开源在线社区论坛作为异步、持久化、多智能体共处的交互媒介（forum-grounded embodied agency），通过标准化的模型上下文协议（MCP）连接线上社区帖子与线下物理机器人。
- **三层系统架构（Section 3 & Figure 1）**：
  1. **Forum Layer**：基于开源论坛平台（仓库说明为 NodeBB）提供 REST API，将板块（boards）、主题（topics）与回复（posts）暴露为可持久化、可检索的多智能体通信底座。
  2. **Agent Layer**：由 TypeScript 编写的 MCP Server（基于 `@modelcontextprotocol/sdk`，通过 JSON-RPC 2.0 通信，仓库亦支持 WebSocket 网络模式）封装 8 个标准化工具（1 个 Meta：`get_manual`；3 个 Read：`list_boards`、`list_posts`、`get_topic`；2 个 Write：`create_topic`、`reply_to_topic`；2 个 Identity：`login_account`、`register_account`），写工具会自动注入 `[agent_id=...]` 等元数据标签以区分人机帖子并防止回帖死循环。Python 侧论坛智能体（`Go2ForumAgent` / `G1ForumAgent`）按固定轮询周期（默认 30 秒）执行 Perceive-Reason-Act 闭环：轮询新帖 → 匹配 `@mention` → LLM 提取指令（失败时降级为规则匹配）→ 调度执行 → LLM 总结结果并回帖。
  3. **Robot Layer**：由 `robot_command_driver` 初始化基于 DDS 的 `unitree_sdk2py` 通信层，并调用对应 VLM 控制器（`Go2VLMController` / `G1VLMController`）运行迭代工具调用循环（`while True` 调用 `VLM.chat(messages, tools)` 直至 `finish_reason != "tool_calls"`），将复杂自然语言指令分解为底层原子动作与感知原语序列。
- **支持硬件与原语（Section 3.4 / 5.4 & README）**：
  - **Unitree Go2**（12-DOF 四足，Ethernet/WiFi DDS）：4 个动作原语（`act_move(direction, value, speed)`、`act_hello()`、`act_heart()`、`act_backflip()`）+ 2 个感知原语（`get_front_image()` 返回 base64、`img_to_volc()` 上传 Volcengine TOS 返回 URL）。
  - **Unitree G1**（23-DOF 人形，Ethernet/WiFi DDS）：2 个动作原语（`act_move(direction, value, speed)`、`act_hello()`）+ 同上 2 个感知原语（USB 摄像头）。
- **论文与仓库细节差异核实**：
  - 论文正文 Section 4.5 列出的三种运行模式为 Polling（默认 30s）、HTTP service mode 与 single-run mode，且 MCP Server 描述聚焦于 stdio；官方 GitHub 仓库 `README.md` 则列出 Polling、Network（WebSocket `ws://host:8765/mcp`）与 Test（`--test-mode` 跳过 `@mention` 检查）三种模式，并明确默认 LLM 后端为火山引擎豆包（`doubao-seed-1-8-251228`），同时说明 TypeScript MCP Server 与 NodeBB 部署单独维护。
  - 论文主要为系统架构与实现报告（10 页，2 图，2 表），Table 1 为与 6 类既有系统的 6 维度定性对比，Table 2 为 8 个 MCP 工具清单；全文未包含定量实验成功率、端到端时延测试或消融实验数据。
- **对 wiki 的映射**：[AgentRob: From Virtual Forum Agents to Hijacked Physical Robots](../../wiki/entities/paper-agentic-2602-13591.md)
