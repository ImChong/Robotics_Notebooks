---
type: entity
project_id: 1348109517-abaqus-agent-skills
code: "https://github.com/1348109517/abaqus-agent-skills"
tags:
  - engineering-tools
  - caa-cfd
  - fea
status: complete
updated: 2026-10-07
summary: "Abaqus Agent Skills（1348109517/abaqus-agent-skills）是提供约 17–19 个可复用 Abaqus 工作流技能与静态契约审计器"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/1348109517-abaqus-agent-skills.md
---

# Abaqus Agent Skills（1348109517/abaqus-agent-skills）

## 一句话定义

Abaqus Agent Skills（1348109517/abaqus-agent-skills）是提供约 17–19 个可复用 Abaqus 工作流技能与静态契约审计器。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FEA | Finite Element Analysis | 有限元分析工作流 |
| FEM | Finite Element Method | 有限元离散求解方法 |
| V&V | Verification and Validation | 数值验证与模型确认 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。提供约 17–19 个可复用 Abaqus 工作流技能与静态契约审计器；强调输入可追溯、稳定命名、先诊断后调试，并区分“求解完成”和“工程正确”。

## 核心原理

**项目自身范围：** Reusable workflow skills and a static contract auditor for Abaqus/FEA agents.

**工作流：** 以契约约束模型输入和执行步骤；通过审计检查命名、路径和输入完整性，再将求解结果交给工程验证。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：提供约 17–19 个可复用 Abaqus 工作流技能与静态契约审计器；强调输入可追溯、稳定命名、先诊断后调试，并区分“求解完成”和“工程正确”。

## 局限与风险

文章提示 Windows 长路径可能导致克隆失败；技能或审计器不能替代结构安全审查、网格敏感性与实验验证。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 Apache-2.0。 

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [FEMIS Skill](./test1card-femis-skill.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/1348109517-abaqus-agent-skills.md)
- [1348109517/abaqus-agent-skills 官方仓库](<https://github.com/1348109517/abaqus-agent-skills>)

## 推荐继续阅读

- [项目 README](<https://github.com/1348109517/abaqus-agent-skills/blob/main/README.md>)