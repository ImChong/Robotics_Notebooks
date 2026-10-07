---
type: entity
project_id: tessalabs-space-freecad-mcp
code: "https://github.com/tessalabs-space/freecad-mcp"
tags:
  - engineering-tools
  - caa-cfd
  - cad
status: complete
updated: 2026-10-07
summary: "FreeCAD MCP（Tessalabs）（tessalabs-space/freecad-mcp）是提供 FreeCAD 工程 MCP，包括参数扫描、绘图/渲染和可选 CAE 交接"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/tessalabs-space-freecad-mcp.md
---

# FreeCAD MCP（Tessalabs）（tessalabs-space/freecad-mcp）

## 一句话定义

FreeCAD MCP（Tessalabs）（tessalabs-space/freecad-mcp）是提供 FreeCAD 工程 MCP，包括参数扫描、绘图/渲染和可选 CAE 交接。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CAD | Computer-Aided Design | 计算机辅助设计与几何建模 |
| MCP | Model Context Protocol | 代理调用 CAD/可视化工具的协议 |
| FEM | Finite Element Method | 结构有限元分析方法 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。提供 FreeCAD 工程 MCP，包括参数扫描、绘图/渲染和可选 CAE 交接；交接可涉及 defeaturing、材料/边界标签以及 Elmer、CalculiX、OpenFOAM、DEM。

## 核心原理

**项目自身范围：** Engineering MCP for FreeCAD with parametric sweeps and optional CAE handoff.

**工作流：** 在 FreeCAD 中构建/扫描参数，再按需整理模型和边界条件并转交 CAE 后端。


**仓库映射说明：** 文章未给出该项目完整的 owner/repository URL；此处按文章中的项目名/描述与公开 GitHub 仓库匹配。引用时请以本页链接的仓库 README 为准。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：提供 FreeCAD 工程 MCP，包括参数扫描、绘图/渲染和可选 CAE 交接；交接可涉及 defeaturing、材料/边界标签以及 Elmer、CalculiX、OpenFOAM、DEM。

## 局限与风险

CAE 交接是可选能力且依赖外部软件；边界标签和缺陷简化需要领域专家确认。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 NOASSERTION。 涉及商业软件时，安装包和有效许可证需由使用者自行提供。

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [FreeCAD MCP（sandraschi）](./sandraschi-freecad-mcp.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/tessalabs-space-freecad-mcp.md)
- [tessalabs-space/freecad-mcp 官方仓库](<https://github.com/tessalabs-space/freecad-mcp>)

## 推荐继续阅读

- [项目 README](<https://github.com/tessalabs-space/freecad-mcp/blob/main/README.md>)