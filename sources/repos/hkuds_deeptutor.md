# DeepTutor（HKUDS/DeepTutor）

> 来源归档（ingest）

- **标题：** DeepTutor: Lifelong Personalized Tutoring
- **类型：** repo / agent-infrastructure / education / rag / tutoring
- **作者 / 组织：** [HKUDS](https://github.com/HKUDS)（香港大学数据智能相关开源组织）；论文作者含 Bingxi Zhao、Jiahao Zhang、Chao Huang 等
- **代码：** <https://github.com/HKUDS/DeepTutor>（**已开源**，Apache-2.0）
- **项目页 / 文档：** <https://deeptutor.info/>
- **PyPI：** `deeptutor`（`pip install -U deeptutor`）
- **容器：** `ghcr.io/hkuds/deeptutor:latest`
- **技术报告：** <https://arxiv.org/abs/2604.26962>（*DeepTutor: Towards Agentic Personalized Tutoring*）
- **技能生态：** [EduHub](https://eduhub.deeptutor.info/)（默认 skill registry）
- **许可：** Apache-2.0（以仓库 `LICENSE` 为准）
- **入库日期：** 2026-08-31
- **一句话说明：** **agent-native 终身个性化辅导工作区**——在统一 capability runtime 下贯通 Chat、测验、研究、可视化、解题、课程、Book、Partner IM 与三层可审计 Memory，并以多引擎 RAG、MCP/CLI Apps 与 EduHub skills 扩展；可 **consult** 本机 Claude Code / Codex / Hermes / OpenClaw 等子代理。

## 开源状态（步骤 2.5）

| 项 | 核查（2026-08-31） |
|----|-------------------|
| **GitHub** | 公开仓 [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor)；主语言 Python；Apache-2.0；README 与 [deeptutor.info](https://deeptutor.info/) 均链到本仓 |
| **项目页** | [deeptutor.info](https://deeptutor.info/) 提供安装向导、能力导览与文档；**Collaborate** 入口在站内设链 |
| **PyPI / 容器** | `pip install deeptutor`；GHCR `ghcr.io/hkuds/deeptutor:latest` 可一键起全栈 Web |
| **论文** | arXiv:2604.26962 注释写明 *Code available at https://github.com/HKUDS/DeepTutor* |
| **CLI-only 包** | `packaging/deeptutor-cli` 可从源码 editable 安装；README 写明 **尚未单独发布到 PyPI**（2026-08-31） |
| **结论** | **已开源**（全栈 Web + CLI + Docker + 论文一致）。可选 RAG/Partner/Matrix 等通过 install extras 或 `DEEPTUTOR_EXTRAS` 扩展。 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [DeepTutor 实体页](../../wiki/entities/deeptutor.md) | 升格：定位、架构、安装形态、与代理宿主边界 |
| [CLI-Anything](../../wiki/entities/cli-anything.md) | 同 HKUDS org；DeepTutor **消费** CLI Apps / skills，CLI-Anything **生成** 专业软件 CLI |
| [Hermes Agent](../../wiki/entities/hermes-agent.md) | **My Agents** 可 live consult Hermes；Hermes 亦可作为常驻 agent OS 对照 |
| [OpenClaw](../../wiki/entities/openclaw.md) | SKILL 宿主之一；`deeptutor skill install clawhub:…` 与 ClawHub 互通 |
| [Agent Reach](../../wiki/entities/agent-reach.md) | 互补：Reach 聚合外网读搜；DeepTutor 内置 web/paper search + RAG + 教学闭环 |
| [Model Context Protocol](../../wiki/concepts/model-context-protocol.md) | DeepTutor 维护 per-account MCP Services store，与内置 tools/capabilities 并列 |
| 机器人学习读者 | 可把 `wiki/`、论文 PDF、课程讲义建成 **Knowledge Center**，用 Quiz / Mastery Path / Book 做 **技术栈自学**（非运动控制栈） |

## README / 架构要点（归纳，2026-08-31）

- **定位：** *Lifelong Personalized Tutoring* — 教育场景下的 **agent-native learning workspace**，非通用 coding agent IDE。
- **统一 runtime：** Chat、Ask Questions、Quiz、Research、Visualize、Solve、Course Study、Mastery Path、Immersive Reading/Watching 共用 **capability + tools** 插件模型与 session context。
- **个性化：** 三层 Memory（L1 traces / L2 summaries / L3 synthesis）+ Memory Graph 溯源；Persona、Question Bank、Notebook、Co-Writer、Book 跨工作流共享。
- **知识：** Knowledge Center 支持 **LlamaIndex、PageIndex、GraphRAG、LightRAG、LightRAG Server、IMA、MarginNote 4、Obsidian** 等引擎；可插拔文档解析（MinerU、PyMuPDF4LLM、Apache Tika、LiteParse…）。
- **子代理：** **My Agents** 可连接并 **consult** Claude Code、Codex、Antigravity、Kimi、opencode、MiMo、**Hermes**、**OpenClaw**、DeepSeek Harness 或 Partner；亦可导入历史会话作只读上下文。
- **Partner：** 持久 IM 伴侣（Feishu、Telegram、Slack、Discord、Matrix… 视 extras），共享同一 `ChatOrchestrator` 大脑，独立 `SOUL.md` 与 workspace。
- **技能：** Agent-Skills 格式；默认 **[EduHub](https://eduhub.deeptutor.info/)**；兼容 **ClawHub**；`deeptutor skill install` 带安全门（verdict、zip 防护、剥离 `always:`）。
- **安装四形态：** PyPI 全栈、`git clone` 源码、Docker 单容器、`packaging/deeptutor-cli` 无 Web UI。
- **配置：** `data/user/settings/` 下 JSON/YAML（`model_catalog.json`、`system.json`、`agents.yaml`…）；工作区可用 `DEEPTUTOR_HOME` 或 `deeptutor start --home` 指定。
- **评测叙事（论文）：** 提出 **TutorBench** 与 profile-driven student simulator；报告个性化指标与通用 agentic reasoning 提升（见论文归档）。

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/deeptutor.md`](../../wiki/entities/deeptutor.md)**
- 项目页归档见 [`sources/sites/deeptutor-info.md`](../sites/deeptutor-info.md)
- 技术报告归档见 [`sources/papers/deeptutor_arxiv_2604_26962.md`](../papers/deeptutor_arxiv_2604_26962.md)
- 交叉更新 CLI-Anything / Hermes / OpenClaw / Agent Reach 等关联页的「关联页面」节
