# PentAGI（vxcontrol/pentagi）

> 来源归档（ingest）

- **标题：** PentAGI — Penetration Testing Artificial General Intelligence
- **类型：** repo / self-hosted multi-agent security testing platform
- **官方代码：** <https://github.com/vxcontrol/pentagi>
- **项目页：** <https://pentagi.com/>
- **许可：** 仓库代码为 MIT；产品 EULA 另列授权使用与合规责任，使用或再分发前应分别核对适用条款。
- **开放状态：** 已开源并可自托管；仓库含 Go 后端、Web 前端、Docker Compose 部署文件及架构/执行/隔离文档。外部模型、搜索服务与可选企业能力可能另有账户、配额或服务条款。
- **入库日期：** 2026-10-13
- **一句话说明：** PentAGI 将自然语言安全评估目标编排成多阶段智能体任务，由不同角色协作研究、规划、执行与汇报，并在 Docker 工作容器内运行安全测试工具。
- **沉淀到 wiki：** 是 → [`wiki/entities/pentagi.md`](../../wiki/entities/pentagi.md)

## 架构要点（官方 README / flow_execution 文档）

- 一个 Flow 可包含多个 Task；Task 再拆为有序 Subtask，并以 Action 记录执行与产物。
- 主协调器与 Generator、Refiner、Pentester、Searcher、Reporter 等专职 agent 协作；规划可根据阶段结果调整，系统保留任务状态与记忆。
- Go 后端管理流程、模型提供方、工具和数据持久化；React 前端提供流程管理与运行观察；PostgreSQL/pgvector 保存任务及向量记忆，可选 Neo4j/Graphiti 做知识图谱。
- 工具执行由 Docker 容器承载；官方文档描述了按 flow 分配容器、可配置网络、资源限制与生命周期清理。容器隔离降低影响面，但部署者仍须核查 Docker daemon/socket、网络出口、挂载、权限、凭据和目标授权范围。
- 仓库当前定位是自主与助手引导的渗透测试平台，不应把营销截图中的“行业覆盖”或社区榜单徽章当作独立的准确率/安全性验证。

## 合规与使用边界

官方 EULA 要求仅在网络所有者明确授权的环境中开展测试，并把任务指令、执行结果、第三方模型与搜索服务的数据处理责任交给使用者。它同时警示勿未经充分测试与验证用于关键基础设施、应急系统或生产环境。开始任何运行前，应把范围限定到自有或书面授权资产，并先在隔离靶场验证部署配置。

## 对 wiki 的映射

- [`wiki/entities/pentagi.md`](../../wiki/entities/pentagi.md)：工具实体、智能体编排、安全边界及机器人系统工程关联。
- [`wiki/concepts/software-security-basics.md`](../../wiki/concepts/software-security-basics.md)：补充授权渗透测试作为研发/云边代码安全评估手段的定位。

## 官方文档入口

- [README](https://github.com/vxcontrol/pentagi/blob/main/README.md)
- [Flow 执行与 agent 角色](https://github.com/vxcontrol/pentagi/blob/main/backend/docs/flow_execution.md)
- [Docker 执行与隔离文档](https://github.com/vxcontrol/pentagi/blob/main/backend/docs/docker.md)
- [MIT License](https://github.com/vxcontrol/pentagi/blob/main/LICENSE)
- [End User License Agreement](https://github.com/vxcontrol/pentagi/blob/main/EULA.md)
