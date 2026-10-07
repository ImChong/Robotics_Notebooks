---
type: entity
project_id: soljourner-claude-engineering-skills
code: "https://github.com/Soljourner/claude-engineering-skills"
tags:
  - engineering-tools
  - caa-cfd
  - engineering
status: complete
updated: 2026-10-07
summary: "Claude Engineering Skills（Soljourner/claude-engineering-skills）是面向机械与航空航天的 100 多个 Skills，重点涵盖流体物性、材料与泵数据库、数值计算包、CAD/仿真对接、单位换算、结构分析和泵设计"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/soljourner-claude-engineering-skills.md
---

# Claude Engineering Skills（Soljourner/claude-engineering-skills）

## 一句话定义

Claude Engineering Skills（Soljourner/claude-engineering-skills）是面向机械与航空航天的 100 多个 Skills，重点涵盖流体物性、材料与泵数据库、数值计算包、CAD/仿真对接、单位换算、结构分析和泵设计。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AI | Artificial Intelligence | 以代理调用工程工具的人工智能能力 |
| CAE | Computer-Aided Engineering | 计算机辅助工程工具与工作流 |
| API | Application Programming Interface | 软件之间调用功能的接口 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。面向机械与航空航天的 100 多个 Skills，重点涵盖流体物性、材料与泵数据库、数值计算包、CAD/仿真对接、单位换算、结构分析和泵设计。

## 核心原理

**项目自身范围：** Collection of Claude skills for mechanical, aerospace, and general engineering tasks.

**工作流：** 将领域知识和流程封装成可路由的技能，再与 ANSYS、OpenFOAM、COMSOL、SolidWorks 等外部软件配合。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：面向机械与航空航天的 100 多个 Skills，重点涵盖流体物性、材料与泵数据库、数值计算包、CAD/仿真对接、单位换算、结构分析和泵设计。

## 局限与风险

自身不含求解器；工程结论依赖输入数据、软件工具与使用者对单位和适用范围的核验。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 MIT。 

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [Awesome AI CAE](./kimimgo-awesome-ai-cae.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/soljourner-claude-engineering-skills.md)
- [Soljourner/claude-engineering-skills 官方仓库](<https://github.com/Soljourner/claude-engineering-skills>)

## 推荐继续阅读

- [项目 README](<https://github.com/Soljourner/claude-engineering-skills/blob/master/README.md>)