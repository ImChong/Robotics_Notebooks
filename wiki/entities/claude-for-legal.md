---
type: entity
project_id: anthropic-claude-for-legal
project: https://claude.com/solutions/legal
code: https://github.com/anthropics/claude-for-legal
tags: [llm-agents, agent-infrastructure, skills, anthropic, legal-tech]
status: complete
updated: 2026-10-16
related:
  - ./anthropic-claude-api-skill.md
  - ./deep-agents.md
  - ./anthropic-frontend-design-skill.md
sources:
  - ../../sources/repos/anthropics-claude-for-legal.md
  - ../../sources/sites/claude-com-solutions-legal.md
summary: "Claude for Legal 是 Anthropic 面向法律工作流的插件集合，将执业领域配置、skills、数据连接器与代理工作流组合起来；其可借鉴点是垂直领域 agent 的模块化交付，法律输出仍需专业人员审阅。"
---

# Claude for Legal

**Claude for Legal** 是 Anthropic 为法律工作流提供的一组开源 Claude 插件与参考配置，覆盖法律团队常见的领域流程，并允许团队围绕自身 playbook 配置工作方式。

## 一句话定义

把 Claude 的通用能力按法律执业领域封装为可安装插件、skills、连接器和代理工作流，让团队从配置过的任务入口开始，而不是每次从空白提示词开始。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MCP | Model Context Protocol | 用于连接外部数据源或工具的协议；仓库提供连接器配置。 |
| DPA | Data Processing Agreement | 隐私插件涉及审阅的数据处理协议。 |
| DSAR | Data Subject Access Request | 隐私工作流中处理数据主体请求的任务类型。 |
| CLAUDE.md | Claude instruction file | 项目用其保存用户或团队的工作配置与指令。 |

## 为什么重要

- **垂直工作流的组织方式：** 仓库按法律领域拆分插件，再在插件内组合技能、配置与连接器。这为通用 agent 如何落到专业任务提供一个具体的模块化案例。
- **从提示词走向可维护配置：** Quickstart 要求先运行 cold-start interview，收集法域、升级规则和团队习惯，并写入工作配置；后续技能读取该配置，减少每次任务重新提供背景的成本。
- **对机器人 agent 的参考范围：** [Claude API Skill](./anthropic-claude-api-skill.md) 侧重模型/API 开发，Claude for Legal 侧重领域流程编排。机器人团队可以借鉴“领域配置 + 可复用任务技能 + 工具连接”的拆分方式，但法律插件本身不能直接当成机器人控制方案。

## 核心结构

| 层次 | 项目中的作用 |
|------|--------------|
| 领域插件 | 按商业、隐私、公司、雇佣、诉讼、监管、AI 治理、知识产权等工作拆分任务入口。 |
| Skills 与 commands | 将常见工作流封装成可调用步骤，例如审阅、分诊、事项跟踪或研究。 |
| Practice profile | 通过初始化访谈记录团队的法域、工作规范和升级条件，供相关技能读取。 |
| Connectors | 连接文档、研究与案件工作流所需的外部数据或服务；具体连接能力随环境和授权而异。 |
| Managed-agent cookbooks | 为需要后台或事件驱动执行的工作流提供参考配置。 |

项目 README 当前将仓库概括为 12 个领域插件、5 份 managed-agent cookbook 和 16 个以上连接器；具体清单会随默认分支变化，应以仓库 README 为准。

## 工程实践

- **安装路径：** 使用 Claude Code 时，按 Quickstart 添加官方 marketplace，安装与工作匹配的插件，重启后运行对应的 cold-start interview。
- **先配置再执行：** 初始化访谈用于收集团队规则和法域背景。没有这类配置时，输出容易退化成通用建议。
- **连接器按需授权：** 先确认需要访问的数据与权限，再配置对应的连接器；仓库代码开放不代表第三方服务或数据访问已获授权。
- **机器人系统的迁移思路：** 可将领域工作流建模为“任务入口、运行配置、工具连接、人工升级规则”几个独立层次；涉及机器人运动的控制、时序与安全仍需由机器人软件栈处理。

## 局限与风险

- **法律输出必须审阅：** 仓库将生成内容定位为供律师审阅的草稿。引文、法域适用性、事实前提和时效性都需要核验，不能据此替代法律意见。
- **数据保密与权限：** 处理合同、案件或个人数据前，要评估连接器授权、数据流向、保密义务及组织的安全策略。
- **连接器依赖外部服务：** 可用性、权限和数据覆盖面依赖所接入的服务，不能将其视为统一或完整的法律数据库。
- **领域不可直接泛化：** 法律流程中的审阅与决策逻辑不等于机器人任务规划或控制策略；可借鉴的是 agent 的配置和工作流组织方法。
- **开源状态：** 官方 GitHub 仓库及其配置文件公开，仓库许可证为 Apache-2.0；商业产品页上的功能、集成和可用性以 Anthropic 当前说明为准。

## 关联页面

- [Claude API Skill](./anthropic-claude-api-skill.md) — Anthropic 官方 API 开发技能，与本项目的领域工作流配置形成互补。
- [Deep Agents](./deep-agents.md) — 开箱 agent harness，可对照其子代理、工具、技能和运行时组织方式。
- [Anthropic frontend-design Skill](./anthropic-frontend-design-skill.md) — 同为 Anthropic 插件/skill 生态中的垂直工作流例子。

## 参考来源

- [Claude for Legal 官方仓库归档](../../sources/repos/anthropics-claude-for-legal.md)
- [Claude 法律行业官方产品页归档](../../sources/sites/claude-com-solutions-legal.md)

## 推荐继续阅读

- [Claude for Legal GitHub README](https://github.com/anthropics/claude-for-legal/blob/main/README.md) — 插件、skills、agent cookbook 与连接器的完整清单。
- [Claude for Legal Quickstart](https://github.com/anthropics/claude-for-legal/blob/main/QUICKSTART.md) — 插件安装与初始化步骤。
