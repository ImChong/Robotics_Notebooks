# OpenResearch（alphaXiv 官方研究 Agent 工作台）

- **类型：** 官方 GitHub 仓库 / 研究代理运行时和工作台
- **代码：** <https://github.com/alphaXiv/OpenResearch>
- **官网：** <https://openresearch.sh/>
- **许可：** MIT（以仓库 LICENSE 为准）
- **研究关联：** [A Society of AI Scientists](../papers/society_ai_scientists_alphaxiv_2611.md)
- **入库日期：** 2026-11-20
- **核对范围：** 官方 README、`src/main.rs`、`src/commands/mod.rs`、`src/commands/up.rs`、`src/commands/create_experiment.rs`、`src/commands/exp.rs` 的 CLI 入口和调用逻辑；未运行程序。
- **通读：** 未完成

## 一句话

OpenResearch 是本地优先的科研 Agent 管理平台，复用现有 coding agent，提供实验节点、运行日志、算力后端与本地工作台；不能将它与论文中的所有 35 个实验室部署环境画等号。

## 官方文档与源码已核查的入口

1. `orx up`：在 `src/main.rs` 解析命令，由 `src/commands/up.rs` 启动本地 Axum API/仪表盘，并初始化 `Store` 和 `Engine`；README 指明默认回环地址 `127.0.0.1:4791`。
2. `orx create-experiment`：`src/commands/create_experiment.rs` 以项目身份、标题、父实验及运行命令构建 `CreateExperimentSpec`，由项目平面接口创建实验节点。
3. `orx exp run`：`src/commands/exp.rs` 将请求交给 `resolve_experiment(...).launch(...)`。
4. 官方 README 还列明本地计算与 SSH、Slurm、Kubernetes 等后端，具体依赖须按部署环境核对。

## 开源状态

官方仓库公开、MIT 许可；存在真实 CLI 运行入口。已核对静态源码调用，但**没有实际启动 `orx` 或完成运行测试**；历史队列的实验脚本和对话记录以官方 Hugging Face 归档为准。

## 对 wiki 的映射

- [论文独立详情页](../../wiki/entities/paper-society-of-ai-scientists.md) —— 含基于以上真实代码入口的 Mermaid 源码时序图。

## 参考来源

- <https://github.com/alphaXiv/OpenResearch/blob/main/README.md>
- <https://github.com/alphaXiv/OpenResearch/blob/main/src/main.rs>
- <https://github.com/alphaXiv/OpenResearch/blob/main/src/commands/up.rs>
- <https://github.com/alphaXiv/OpenResearch/blob/main/src/commands/create_experiment.rs>
- <https://github.com/alphaXiv/OpenResearch/blob/main/src/commands/exp.rs>
