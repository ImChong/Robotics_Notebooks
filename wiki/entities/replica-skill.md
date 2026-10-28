---
type: entity
project_id: replica-skill
code: https://github.com/Jakeschincariol/replica-skill
tags:
  - llm-agents
  - coding-agents
  - skills
  - agent-infrastructure
  - software-engineering
status: complete
updated: 2026-10-27
related:
  - ./agent-skills-addyosmani.md
  - ./anthropic-claude-api-skill.md
  - ../concepts/agentic-coding-software-fundamentals.md
sources:
  - ../../sources/repos/replica-skill.md
summary: "Replica Skill 是 11 个 Claude Skills 组成的应用重建工作流：从界面与功能调研、架构设计、实现和测试，到差异评分、品牌重塑与部署；适合原型开发参考，不是机器人控制栈。"
---

# Replica Skill（app 克隆 Agent Skills）

**Replica Skill** 是一组面向 Claude 的 11 个 Agent Skills，按从应用调研到上线的顺序，将目标应用的功能和交互整理成可执行的重建计划，并在用户自己的项目中实现。其 README 明确要求 clean-room 重写，强调重建功能与流程，不复制目标应用的代码、商标、文案或专有内容。

## 一句话定义

它把应用复刻工作拆分为 11 个阶段技能和配套检查工具，让编码代理基于调研产物逐步构建并检查一个独立实现。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| UI | User Interface | 调研与重建的应用界面 |
| UX | User Experience | 工作流关注的交互体验与任务流程 |
| MIT | Massachusetts Institute of Technology License | 仓库使用的宽松开源许可证 |

## 为什么重要（对本知识库读者）

- **适合做机器人周边软件原型：** 若团队要快速构建设备管理面板、数据标注界面或实验室工具，分阶段拆解需求、界面和测试的做法可作为 coding-agent 工作流参考。
- **与通用编码技能包侧重点不同：** [Agent Skills（Addy Osmani）](agent-skills-addyosmani.md)强调软件开发生命周期与工程质量门禁；Replica Skill 聚焦将一个既有应用的功能与交互转化为新实现，并覆盖产品定位和发布准备。
- **技术边界清晰：** 仓库不是机器人 SDK、仿真器或控制框架，不提供运动控制算法、机器人模型、策略或硬件接口。

## 核心流程

| 阶段 | 技能 | 产物或作用 |
|------|------|------------|
| 调研与规划 | `replica-recon`、`replica-architect` | 页面、流程、组件、数据模型及技术栈方案 |
| 设计与实现 | `replica-design`、`replica-build`、`replica-backend` | 设计 token、逐屏实现、认证/数据库/支付等后端接线 |
| 验证 | `replica-test`、`replica-diff` | 测试计划与缺陷记录、功能覆盖与截图差异评估 |
| 产品化 | `replica-entrepreneur`、`replica-brand`、`replica-launch`、`replica-deploy` | 用户评论分析、重新命名与品牌、商店/落地页文案和部署 |

各阶段通过项目内的 `replica/` 目录共享调研与实现资料。README 还列出 6 个标准库 Python 工具，用于像素边缘差异、功能矩阵评分、评论主题统计、对比度、品牌残留和商店文案检查；工具不代替人工审核或真实用户测试。

## 工程实践

- **安装：** README 提供 Claude 中直接安装、Claude Code 插件安装，以及将技能目录复制到全局或项目级 `.claude/skills/` 的方式。
- **依赖：** 所列工具使用 Python 标准库；要求 Python 3.8 或更新版本。
- **适用场景：** 需要快速探索竞品公开功能、制作独立原型、完善端到端测试与上线清单的产品工程任务。
- **开源状态：** 已开源，许可证为 MIT。该仓库未提供机器人专用实现。

## 局限与风险

- 截图差异分数和功能矩阵只能衡量预先定义的维度，不能证明新应用在可用性、安全性或可靠性上等同于目标产品。
- 项目方的 clean-room 与访问边界是工作流约束；使用者仍需核对目标服务条款、素材权利和商标，并避免复制私有数据或访问控制外内容。
- `replica-entrepreneur` 依赖带来源链接的用户评论数据；结论质量受样本覆盖和来源偏差影响。
- 对机器人系统，应将该流程限制在非安全关键的界面或开发工具原型；实时控制、急停、安全策略和设备协议需按真实系统要求独立设计和验证。

## 关联页面

- [Agent Skills（Addy Osmani）](agent-skills-addyosmani.md) — 覆盖完整软件开发生命周期的技能包，可与本项目对照。
- [claude-api（Anthropic 官方 Skill）](anthropic-claude-api-skill.md) — 面向 Claude API 集成、评测和成本优化的官方技能。
- [Agentic Coding 时代的软件工程基础](../concepts/agentic-coding-software-fundamentals.md) — 使用编码代理时仍需明确规格、验证和工程责任。

## 参考来源

- [Replica Skill 仓库归档](../../sources/repos/replica-skill.md)
- [Jakeschincariol/replica-skill README](https://github.com/Jakeschincariol/replica-skill/blob/main/README.md)
- [Jakeschincariol/replica-skill LICENSE](https://github.com/Jakeschincariol/replica-skill/blob/main/LICENSE)

## 推荐继续阅读

- [Jakeschincariol/replica-skill](https://github.com/Jakeschincariol/replica-skill) — 安装说明、11 项技能定义、工具脚本与边界说明。
