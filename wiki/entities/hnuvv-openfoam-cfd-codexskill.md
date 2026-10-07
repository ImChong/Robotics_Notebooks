---
type: entity
project_id: hnuvv-openfoam-cfd-codexskill
code: "https://github.com/HNUVV/openfoam-CFD-codexskill"
tags:
  - engineering-tools
  - caa-cfd
  - cfd
status: complete
updated: 2026-10-07
summary: "OpenFOAM CFD Codex Skill（HNUVV/openfoam-CFD-codexskill）是文章将它作为中文 Codex Skill 入门模板"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/hnuvv-openfoam-cfd-codexskill.md
---

# OpenFOAM CFD Codex Skill（HNUVV/openfoam-CFD-codexskill）

## 一句话定义

OpenFOAM CFD Codex Skill（HNUVV/openfoam-CFD-codexskill）是文章将它作为中文 Codex Skill 入门模板。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CFD | Computational Fluid Dynamics | 计算流体力学计算流程 |
| MCP | Model Context Protocol | 让代理发现并调用外部工具的协议 |
| y+ | Dimensionless Wall Distance | 壁面网格分辨率相关的无量纲距离 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。文章将它作为中文 Codex Skill 入门模板；仓库简介表明它提供 CFD 辅助技能说明，适合观察一个小型、单主题 Skill 的结构。

## 核心原理

**项目自身范围：** A Codex skill for computational fluid dynamics.

**工作流：** 将 CFD 任务约束和 OpenFOAM 相关步骤写入代理可读取的技能文件，再由用户在已有求解器环境中执行。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：文章将它作为中文 Codex Skill 入门模板；仓库简介表明它提供 CFD 辅助技能说明，适合观察一个小型、单主题 Skill 的结构。

## 局限与风险

文章没有给出更细的验证数据；作为入门模板阅读时应自行检查边界、许可证、测试案例和版本覆盖。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 未报告 SPDX 许可证，应检查仓库内 LICENSE 与依赖许可。 

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [OpenFOAM Agent Skills](./twj011-openfoam-agent-skills.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/hnuvv-openfoam-cfd-codexskill.md)
- [HNUVV/openfoam-CFD-codexskill 官方仓库](<https://github.com/HNUVV/openfoam-CFD-codexskill>)

## 推荐继续阅读

- [项目 README](<https://github.com/HNUVV/openfoam-CFD-codexskill/blob/main/README.md>)