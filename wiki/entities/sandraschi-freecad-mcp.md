---
type: entity
project_id: sandraschi-freecad-mcp
code: "https://github.com/sandraschi/freecad-mcp"
tags:
  - engineering-tools
  - caa-cfd
  - cad
status: complete
updated: 2026-10-07
summary: "FreeCAD MCP（sandraschi）（sandraschi/freecad-mcp）是通过 FastMCP 自动化 FreeCAD，支持 headless 文档/导出，并扩展 FluidX3D 与 OpenFOAM CFD"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/sandraschi-freecad-mcp.md
---

# FreeCAD MCP（sandraschi）（sandraschi/freecad-mcp）

## 一句话定义

FreeCAD MCP（sandraschi）（sandraschi/freecad-mcp）是通过 FastMCP 自动化 FreeCAD，支持 headless 文档/导出，并扩展 FluidX3D 与 OpenFOAM CFD。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CAD | Computer-Aided Design | 计算机辅助设计与几何建模 |
| MCP | Model Context Protocol | 代理调用 CAD/可视化工具的协议 |
| FEM | Finite Element Method | 结构有限元分析方法 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。通过 FastMCP 自动化 FreeCAD，支持 headless 文档/导出，并扩展 FluidX3D 与 OpenFOAM CFD；这是独立仓库，不等同于知识库已有的 neka-nat/freecad-mcp。

## 核心原理

**项目自身范围：** FreeCAD automation via FastMCP with headless export and CFD extensions.

**工作流：** MCP 服务器与 FreeCAD 文档/导出交互，并可将任务连接到 FluidX3D 或 OpenFOAM 扩展流程。


**仓库映射说明：** 文章未给出该项目完整的 owner/repository URL；此处按文章中的项目名/描述与公开 GitHub 仓库匹配。引用时请以本页链接的仓库 README 为准。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：通过 FastMCP 自动化 FreeCAD，支持 headless 文档/导出，并扩展 FluidX3D 与 OpenFOAM CFD；这是独立仓库，不等同于知识库已有的 neka-nat/freecad-mcp。

## 局限与风险

外部扩展求解器仍需安装和验证；与其他同名 freecad-mcp 仓库要以所有者/URL 区分。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 MIT。 涉及商业软件时，安装包和有效许可证需由使用者自行提供。

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [FreeCAD MCP（Tessalabs）](./tessalabs-space-freecad-mcp.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/sandraschi-freecad-mcp.md)
- [sandraschi/freecad-mcp 官方仓库](<https://github.com/sandraschi/freecad-mcp>)

## 推荐继续阅读

- [项目 README](<https://github.com/sandraschi/freecad-mcp/blob/master/README.md>)