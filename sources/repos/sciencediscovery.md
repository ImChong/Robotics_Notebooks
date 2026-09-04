# ScienceDiscovery（代码仓库）

> 来源归档

- **标题：** ScienceDiscovery — 一站式 AI 科研工作台
- **类型：** repo
- **组织：** [openJiuwen-ai](https://github.com/openJiuwen-ai)（华为 2012 实验室 / 华为云 / 终端 / 计算共建的开源 Agent 社区）
- **GitHub：** <https://github.com/openJiuwen-ai/sciencediscovery>
- **AtomGit：** <https://atomgit.com/openJiuwen/sciencediscovery>
- **官网 / 平台：** <https://www.openjiuwen.com>
- **许可：** Apache License 2.0
- **主语言：** TypeScript（pnpm monorepo）+ Python sidecar（MCP / PDF worker / Gateway 解释器环境）
- **入库日期：** 2026-09-04
- **一句话说明：** 面向科学研究的本地单用户 AI 工作台：文献 Connector、Bubblewrap 沙箱跑 Python/R/Shell、Node 内原生 agent 环拆解任务，产物走 CAS 溯源；默认只监听回环、单 bearer token、无 TLS。
- **沉淀到 wiki：** [`wiki/entities/sciencediscovery.md`](../../wiki/entities/sciencediscovery.md)；平台总览见 [`wiki/entities/openjiuwen.md`](../../wiki/entities/openjiuwen.md)

## 开源状态（步骤 2.5）

**已开源**：GitHub 与 AtomGit 双托管同一产品仓；Apache-2.0；文档树完整（中英 Tutorial / How-to / Reference / Explanation）。模型权重不随仓分发——产品自述「仅作流程编排工具，不内嵌模型能力」，用户自行配置 provider。

| 通道 | URL | 备注 |
|------|-----|------|
| GitHub（文档与 API 较完整） | <https://github.com/openJiuwen-ai/sciencediscovery> | org `openJiuwen-ai`；截至 2026-09-04 约 48 star、10 fork、TypeScript |
| AtomGit | <https://atomgit.com/openJiuwen/sciencediscovery> | 中文简介与 GitHub README_zh 对齐 |
| 项目页 | 无独立 `*.github.io`；产品入口走 [openJiuwen 官网](https://www.openjiuwen.com) | 官网主推 WorkSwarm / 模型广场，ScienceDiscovery 以代码仓为准 |
| 数据 / 权重 | 不随仓 | 文献与数据库经 MCP Connector 远程查询；科学环境用固定 micromamba，首次创建需包通道 |

## 仓库树（文档 `repository-layout` 归纳）

```text
sciencediscovery/
├── apps/web/                 # React/Vite 工作台
├── services/
│   ├── api/                  # Node 控制面 + native-agent 环（:4310）
│   ├── gateway/              # 不再是常驻服务：bundled Python MCP 的解释器环境
│   ├── runner/               # Bubblewrap 执行器（:4311，仅回环）
│   ├── paper/                # 按次拉起的 PDF worker
│   └── memory-graph/         # 实验性 Science Memory，默认关
├── packages/                 # agent-runtime / schema / mcp-sources
├── skills/                   # 仓内可见内置包（生命科学 brief、PDB pocket）
├── docs/                     # 中英完整文档
└── LICENSE                   # Apache-2.0
```

默认常驻进程 **两个**：Runner `:4311` + API `:4310`（agent 环、模型调用、进程内 MCP client 都在 API 内）。`services/gateway` 只给 biomed / UniProt 等 Python MCP 提供 stdio 解释器。

## README 要点（归纳，非转存）

- **定位：** 浏览器 UI ↔ Node 控制 API；每个 run 由 **本仓 TypeScript `native-agent/`** 驱动（文档明确 **不用 LangChain / LangGraph**）。工作区工具、沙箱、科学 Connector、PDF、权限、provenance、审阅由 Node 控制面强制执行。
- **安全边界（WARNING）：** 不是多用户生产服务。API / runner 默认 loopback；单 bearer token、不终止 TLS。Python / R / shell 在 fail-closed bubblewrap 里跑；控制 API、gateway 解释器、PDF worker、出站模型/数据请求在沙箱外作为受信任控制面。
- **产品 README 宣称的四条能力：** 科研数据库 Connector 一键接入；沙箱自主写/调/跑代码；任务规划与多智能体协同、宣称动态编排 300+ Skills；代码/环境/日志全链路可视化溯源。
- **Skills 口径差：** 仓内 `skills/` 文档只列 **2** 个内置包（`life-science-evidence-brief`、`structure-pocket-inspection`）。「300+」应读成 **openJiuwen / SkillHub 生态叙事或后续扩展**，选型以 Session 启用清单与 `describe_skill` 为准，不要把 README 营销句当成仓内技能目录长度。
- **部署三路径互斥：** 预打包单文件二进制（运行时几乎只依赖 host `bwrap`）、源码模式（Node 22.19+ / pnpm 11.1.2 / Python 3.12 via uv / bubblewrap）、Docker Compose。全部要求 Linux x86_64 或 aarch64 + 无特权 user namespace。**不支持把二进制再塞进镜像。**
- **最短路径：** `./ScienceDiscovery serve` → `curl` `:4310` health → 用启动打印的 token 登录 → **系统配置 → Global defaults** 配任务模型。
- **Science Memory** 需要外部 Neo4j，未配置则保持关闭，不影响 Web/对话。
- **昇腾：** `SCIENCE_AGENT_NPU_BROKER=1` 时 Runner 暴露 allowlist 内的 `run_npu_job`（内置 `npu.smoke_test`、`antibody.protenix.v1`），不是通用宿主机 shell。

## 默认端口与工具面

| 地址 | 职责 |
|------|------|
| `127.0.0.1:4310` | 控制 API + 静态 Web UI + **进程内 agent 环** |
| `127.0.0.1:4311` | Runner（仅回环） |
| 无端口 | gateway venv：stdio MCP 解释器 |

内置工具（模型可见）包括：`list_files` / `read_file` / Artifact 声明、`run_python` / `run_shell`（及托管环境下的 `run_r`）、`web_search` / `web_fetch`、主 run 的 `propose_plan` / `task`（子代理，每请求最多 50 个 child、每轮最多 10 次 `task`）、延迟披露的 `mcp__<source>__<tool>`、`artifact_download` + `paper_extract_pdf`（必须分 turn）。文献源：PubMed、arXiv、Europe PMC、bioRxiv、medRxiv；库：UniProt、PDB、Ensembl、Reactome、ClinVar、ChEMBL、GEO。

## 对 wiki 的映射

- 升格实体页：[ScienceDiscovery](../../wiki/entities/sciencediscovery.md)
- 平台页：[openJiuwen](../../wiki/entities/openjiuwen.md)
- 交叉：
  - [AI Auto-Research](../../wiki/concepts/ai-auto-research.md) — S2 文献 + S3 沙箱实验的工作台形态
  - [karpathy/autoresearch](../../wiki/entities/karpathy-autoresearch.md) — 对照：固定 metric 最小实验环 vs 开放文献/代码工作台
  - [Hermes Agent](../../wiki/entities/hermes-agent.md) / [OpenClaw](../../wiki/entities/openclaw.md) / [DeepSeek Harness](../../wiki/entities/deepseek-harness.md) — 通用 agent OS 对照
  - [Model Context Protocol](../../wiki/concepts/model-context-protocol.md) — 科学 Connector 的治理链（Node broker + 进程内 client + Python MCP）

## 参考来源（原始）

- GitHub README / README_zh（2026-09-04 抓取）
- `docs/en/explanation/architecture.md`、`control-plane.md` 对等 Explanation 索引
- `docs/en/how-to/deployment.md`、`docs/en/tutorial/01-quick-start.md`
- `docs/en/reference/repository-layout.md`、`builtin-tools.md`
- AtomGit 项目简介：<https://atomgit.com/openJiuwen/sciencediscovery>
