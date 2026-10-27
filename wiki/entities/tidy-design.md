---
type: entity
project_id: tidy-design
tags:
  - llm-agents
  - coding-agents
  - mcp
  - design
  - software-engineering
  - open-source
status: complete
updated: 2026-10-19
project: https://tidydesign.co
code: https://github.com/tidydesigns/tidy
related:
  - ../concepts/model-context-protocol.md
  - ../concepts/agentic-coding-software-fundamentals.md
  - ./anthropic-frontend-design-skill.md
  - ./freecad-mcp.md
sources:
  - ../../sources/sites/tidy-design.md
  - ../../sources/repos/tidydesigns-tidy.md
summary: "Tidy 是面向设计师、工程师和 agent 的开源产品设计画布，通过 MCP 让 agent 读取、创建和编辑界面设计；可作为设计与编码协作工具观察，不是机器人 CAD 或仿真工具。"
---

# Tidy Design：面向智能体的可编辑产品画布

## 一句话定义

Tidy 是面向设计师、工程师和 agent 的开源产品画布：人可直接编辑界面，兼容 MCP 的 agent 可读取、创建和修改设计，团队能在同一工作区协作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MCP | Model Context Protocol | Tidy 用于连接 agent 与设计画布的工具协议 |
| UI | User Interface | 画布中创建和迭代的产品界面 |
| API | Application Programming Interface | 自托管部署连接数据库、存储等服务的接口边界 |

## 为什么重要

- **设计产物保持可编辑：** Tidy README 强调 agent 与人操作同一设计文件和图层，便于从生成稿继续修改，而不只交换扁平截图。
- **连接设计与编码协作：** 设计画布可承担 coding agent 的上下文和评审界面；这不等于自动把设计无误转成生产代码。
- **对机器人研发的潜在价值：** 可探索用于设备状态面板、遥操作界面、仿真分析仪表盘原型。这是基于产品能力的推论，官方仓库没有展示机器人专用集成或验证。

## 核心原理

| 能力层 | 官方资料确认的内容 | 使用上的理解 |
|----------|------------------|--------------|
| 人类画布 | 在画布上创建界面，图层保持可编辑 | 人可直接调整 agent 生成的设计 |
| Agent 接口 | 通过 MCP 读取、创建和编辑设计 | agent 可在同一工作区迭代设计 |
| 团队协作 | README 描述同一工作区协作 | 设计、工程与 agent 共享工作产物 |
| 网页采集 | 浏览器扩展把网页带入 Tidy | 可把现有页面作为参考再调整 |

仓库提供 agent plugin 配置指南和 native runner 指南。具体客户端支持与安装步骤以这些一手文档为准。

## 工程实践

- **先把它当设计工作区：** 配置仓库提供的 agent 插件，再让 agent 针对同一设计进行创建或修改；提交前由设计师或工程师检查层级、文案、布局与状态。
- **本地试用：** README 列出 Bun 1.3.14、Git 和 Docker Compose；新建本地环境后按环境变量示例配置并启动。自托管涉及数据库、密钥和存储绑定，需按完整文档操作。
- **机器人场景先验证：** 可从非安全关键的内部面板原型开始，验证设计编辑、人员复核与导出交接。不能假定具备 CAD 几何精度、URDF/MJCF 导出或仿真器接口。

## 局限与风险

- **不是机器人 CAD 或仿真工具：** 已核查的 README 定位为产品设计画布，没有展示 URDF、MJCF、机器人运动学或仿真功能。
- **MCP 只说明工具接入：** 协议可读写设计，并不自动保证视觉质量、可访问性、代码正确性或权限边界。
- **官网内容未独立核验：** 官网入口由官方 README 确认，但官网页面本身无法读取；功能和许可判断基于公开仓库。
- **自托管需运维配置：** README 的 Cloudflare Worker 部署涉及数据库、存储绑定和运行密钥，并非零配置部署。

## 关联页面

- [Desigma](./desigma.md) — Rust 本地优先设计工具，也提供 MCP 接口。
- [Model Context Protocol](../concepts/model-context-protocol.md) — agent 与外部工具交互的协议层。
- [Agentic Coding 时代的软件工程基础](../concepts/agentic-coding-software-fundamentals.md) — 设计协作仍需架构、约束和评测判断。
- [frontend-design（Anthropic 官方 Skill）](./anthropic-frontend-design-skill.md) — 设计规约与视觉判断的补充工具。
- [FreeCAD MCP](./freecad-mcp.md) — MCP 驱动专业桌面软件的机器人 CAD 对照案例。

## 参考来源

- [Tidy Design 官网归档](../../sources/sites/tidy-design.md)
- [Tidy 官方仓库归档](../../sources/repos/tidydesigns-tidy.md)
- [Tidy README](https://github.com/tidydesigns/tidy#readme)

## 推荐继续阅读

- [Tidy GitHub 仓库](https://github.com/tidydesigns/tidy) — 源码、开发说明与 agent 接入指南。
- [Tidy 官网](https://tidydesign.co) — 官方产品入口。
- [Model Context Protocol](../concepts/model-context-protocol.md) — 核对 MCP 工具连接、部署与权限边界。
