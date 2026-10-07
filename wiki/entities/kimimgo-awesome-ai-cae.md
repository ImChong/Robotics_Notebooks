---
type: entity
project_id: kimimgo-awesome-ai-cae
code: "https://github.com/kimimgo/awesome-ai-cae"
tags:
  - engineering-tools
  - cae-cfd
  - engineering
status: complete
updated: 2026-10-07
summary: "Awesome AI CAE（kimimgo/awesome-ai-cae）是收集 110 多个可供 AI 调用的 CAE 工具，横跨 CFD、FEA、SPH、DEM、网格、CAD 与可视化"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/kimimgo-awesome-ai-cae.md
---

# Awesome AI CAE（kimimgo/awesome-ai-cae）

## 一句话定义

Awesome AI CAE（kimimgo/awesome-ai-cae）是收集 110 多个可供 AI 调用的 CAE 工具，横跨 CFD、FEA、SPH、DEM、网格、CAD 与可视化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AI | Artificial Intelligence | 以代理调用工程工具的人工智能能力 |
| CAE | Computer-Aided Engineering | 计算机辅助工程工具与工作流 |
| API | Application Programming Interface | 软件之间调用功能的接口 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。收集 110 多个可供 AI 调用的 CAE 工具，横跨 CFD、FEA、SPH、DEM、网格、CAD 与可视化；以 MCP、Python API、CLI 等可调用性维度排序，而非以 star 数作为主要依据。

## 核心原理

**项目自身范围：** Curated list of 100+ AI-ready tools for CAE, ranked by agent-callability.

**工作流：** 先按求解域筛选工具，再核对它是否有 MCP、Python API 或 CLI；随后才评估具体工程能力。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：收集 110 多个可供 AI 调用的 CAE 工具，横跨 CFD、FEA、SPH、DEM、网格、CAD 与可视化；以 MCP、Python API、CLI 等可调用性维度排序，而非以 star 数作为主要依据。

## 局限与风险

这是策展目录，不提供求解器实现，也不保证收录工具版本、许可证或能力始终同步。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 CC0-1.0。 

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [HPC-Skills](./scimate-ai-hpc-skills.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/kimimgo-awesome-ai-cae.md)
- [kimimgo/awesome-ai-cae 官方仓库](<https://github.com/kimimgo/awesome-ai-cae>)

## 推荐继续阅读

- [项目 README](<https://github.com/kimimgo/awesome-ai-cae/blob/main/README.md>)