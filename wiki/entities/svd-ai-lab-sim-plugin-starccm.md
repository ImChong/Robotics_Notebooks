---
type: entity
project_id: svd-ai-lab-sim-plugin-starccm
code: "https://github.com/svd-ai-lab/sim-plugin-starccm"
tags:
  - engineering-tools
  - svd-ai-lab
  - caa-cfd
  - mcp
status: complete
updated: 2026-10-07
summary: "sim-plugin-starccm（svd-ai-lab/sim-plugin-starccm）是sim-cli 的 STAR-CCM+ 外置驱动，使用 Java StarMacro 通过 `starccm+ -batch` 执行"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/svd-ai-lab-sim-plugin-starccm.md
---

# sim-plugin-starccm（svd-ai-lab/sim-plugin-starccm）

## 一句话定义

sim-plugin-starccm（svd-ai-lab/sim-plugin-starccm）是sim-cli 的 STAR-CCM+ 外置驱动，使用 Java StarMacro 通过 `starccm+ -batch` 执行。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MCP | Model Context Protocol | 代理与工程软件工具的交互协议 |
| CAE | Computer-Aided Engineering | 计算机辅助工程软件与分析 |
| API | Application Programming Interface | 软件可供自动化调用的接口 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。sim-cli 的 STAR-CCM+ 外置驱动，使用 Java StarMacro 通过 `starccm+ -batch` 执行；插件同时注册驱动和技能目录，并含兼容性矩阵。

## 核心原理

**项目自身范围：** STAR-CCM+ driver plugin for sim-cli, using Java macros in batch mode.

**工作流：** 代理/CLI 生成或调用宏脚本，驱动以批处理运行 STAR-CCM+，再把场景与结果检查信息汇回工作流。


**仓库映射说明：** 文章未给出该项目完整的 owner/repository URL；此处按文章中的项目名/描述与公开 GitHub 仓库匹配。引用时请以本页链接的仓库 README 为准。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：sim-cli 的 STAR-CCM+ 外置驱动，使用 Java StarMacro 通过 `starccm+ -batch` 执行；插件同时注册驱动和技能目录，并含兼容性矩阵。

## 局限与风险

STAR-CCM+ 没有可 pip 安装的 SDK；软件和授权必须用户提供，端到端测试也要求主机真实安装。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 Apache-2.0。 涉及商业软件时，安装包和有效许可证需由使用者自行提供。

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [AutoStar](./ouscar-ou-autostar.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/svd-ai-lab-sim-plugin-starccm.md)
- [svd-ai-lab/sim-plugin-starccm 官方仓库](<https://github.com/svd-ai-lab/sim-plugin-starccm>)

## 推荐继续阅读

- [项目 README](<https://github.com/svd-ai-lab/sim-plugin-starccm/blob/main/README.md>)