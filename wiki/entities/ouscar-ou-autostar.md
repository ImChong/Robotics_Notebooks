---
type: entity
project_id: ouscar-ou-autostar
code: "https://github.com/Ouscar-ou/AutoStar"
tags:
  - engineering-tools
  - caa-cfd
  - cfd
status: complete
updated: 2026-10-07
summary: "AutoStar（Ouscar-ou/AutoStar）是针对 STAR-CCM+ 螺旋桨敞水 CFD"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/ouscar-ou-autostar.md
---

# AutoStar（Ouscar-ou/AutoStar）

## 一句话定义

AutoStar（Ouscar-ou/AutoStar）是针对 STAR-CCM+ 螺旋桨敞水 CFD。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CFD | Computational Fluid Dynamics | 计算流体力学计算流程 |
| MCP | Model Context Protocol | 让代理发现并调用外部工具的协议 |
| y+ | Dimensionless Wall Distance | 壁面网格分辨率相关的无量纲距离 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。针对 STAR-CCM+ 螺旋桨敞水 CFD。重点约束 STEP/STP 单位、几何包围盒、桨轴/来流/进出口与旋转方向；先 quick/coarse 验流程，再用 400 步 pilot 检查残差、推力、扭矩和 y+。

## 核心原理

**项目自身范围：** STAR-CCM+ propeller open-water CFD workflow skill.

**工作流：** 执行几何预检、方向确认、粗网格试算、pilot 稳定性筛查，通过后才继续计算，并保存 preflight 与运行报告。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：针对 STAR-CCM+ 螺旋桨敞水 CFD。重点约束 STEP/STP 单位、几何包围盒、桨轴/来流/进出口与旋转方向；先 quick/coarse 验流程，再用 400 步 pilot 检查残差、推力、扭矩和 y+。

## 局限与风险

STAR-CCM+ 及其许可证由使用者提供；400 步筛查不是最终收敛证明，需按目标工况补充网格与模型验证。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 NOASSERTION。 涉及商业软件时，安装包和有效许可证需由使用者自行提供。

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [sim-plugin-starccm](./svd-ai-lab-sim-plugin-starccm.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/ouscar-ou-autostar.md)
- [Ouscar-ou/AutoStar 官方仓库](<https://github.com/Ouscar-ou/AutoStar>)

## 推荐继续阅读

- [项目 README](<https://github.com/Ouscar-ou/AutoStar/blob/main/README.md>)