---
type: entity
project_id: v0v1kkk-freecad-engineering
code: "https://github.com/V0v1kkk/freecad-engineering"
tags:
  - engineering-tools
  - caa-cfd
  - cad
status: complete
updated: 2026-10-07
summary: "FreeCAD Engineering（V0v1kkk/freecad-engineering）是提供可编辑、带证据的 FreeCAD 工程技能与 MCP 配置，强调每个建模步骤保留可检查依据"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/v0v1kkk-freecad-engineering.md
---

# FreeCAD Engineering（V0v1kkk/freecad-engineering）

## 一句话定义

FreeCAD Engineering（V0v1kkk/freecad-engineering）是提供可编辑、带证据的 FreeCAD 工程技能与 MCP 配置，强调每个建模步骤保留可检查依据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CAD | Computer-Aided Design | 计算机辅助设计与几何建模 |
| MCP | Model Context Protocol | 代理调用 CAD/可视化工具的协议 |
| FEM | Finite Element Method | 结构有限元分析方法 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。提供可编辑、带证据的 FreeCAD 工程技能与 MCP 配置，强调每个建模步骤保留可检查依据；与直接文本生成模型区分开。

## 核心原理

**项目自身范围：** Editable, evidence-backed FreeCAD engineering skills and MCP configuration.

**工作流：** 代理按工程技能执行参数化 CAD 操作，同时保留模型修改与依据，便于审阅设计意图。


**仓库映射说明：** 文章未给出该项目完整的 owner/repository URL；此处按文章中的项目名/描述与公开 GitHub 仓库匹配。引用时请以本页链接的仓库 README 为准。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：提供可编辑、带证据的 FreeCAD 工程技能与 MCP 配置，强调每个建模步骤保留可检查依据；与直接文本生成模型区分开。

## 局限与风险

证据链针对过程透明度，不自动证明设计通过强度、制造或法规审核。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 MIT。 涉及商业软件时，安装包和有效许可证需由使用者自行提供。

## 源码运行时序图

入口：Agent 客户端按 `plugin.json` / `mcp.json` 启动 `freecad-mcp`（外部 PyPI 包，不在本仓库内）；本仓库自带的可运行入口为 `python skills/freecad-engineering-bootstrap/scripts/bootstrap_probe.py`、`python skills/freecad-engineering-bootstrap/scripts/render_mcp_config.py` 与 `npm --prefix scripts/viewer run start -- --dir <目录>`（位于 `skills/cad-viewer/`）。

```mermaid
sequenceDiagram
    participant U as 用户
    participant A as Agent 客户端
    participant S as freecad-engineering/SKILL.md
    participant P as bootstrap_probe.py（只读探测）
    participant R as render_mcp_config.py
    participant M as freecad-mcp（外部 stdio 服务）
    participant F as FreeCAD + MCP 插件（外部）
    participant V as cad-viewer backend/server.mjs
    participant C as cadpy.step_artifact

    U->>A: 提出小型机械零件建模需求
    A->>S: 加载技能，先写需求台账与验收条件
    Note over S: 仅环境或能力诊断时才转入 bootstrap 技能
    A->>P: python scripts/bootstrap_probe.py
    P-->>A: 工具路径、FreeCAD UserAppData、Addon 检测结果
    A->>R: 渲染 MCP 配置（只有给出 --output 才写文件）
    R-->>A: JSON / TOML / codex 命令
    A->>M: 以 stdio 启动 freecad-mcp（需在 PATH 上）
    M->>F: 通过 FreeCAD RPC 读取或执行 Python
    F-->>M: 文档状态与执行结果
    M-->>A: 返回 MCP 工具结果
    A->>S: 计划阶段：Spreadsheet、Sketch、PartDesign 特征、检查点
    alt 需要审阅 STEP / GLB 等产物
        A->>V: node backend/server.mjs --dir <模型目录>（默认 127.0.0.1:4178）
        V-->>A: 监听地址，返回 ?dir=&file= 审阅链接
        V->>C: 需要时 spawn python -m cadpy.step_artifact
        C-->>V: 生成 GLB 与拓扑产物
    end
    A-->>U: 模型文件、校验证据与未测试项
```

图依据：V0v1kkk/freecad-engineering@6d48362，实际读过 `README.md`、`plugin.json`、`mcp.json`、`skills/freecad-engineering/SKILL.md`、`skills/freecad-engineering-bootstrap/SKILL.md`、`skills/freecad-engineering-bootstrap/scripts/bootstrap_probe.py`、`skills/freecad-engineering-bootstrap/scripts/render_mcp_config.py`、`skills/cad-viewer/SKILL.md`、`skills/cad-viewer/scripts/viewer/package.json`、`skills/cad-viewer/scripts/viewer/backend/server.mjs`（bundled 文件，仅读取参数解析、监听与 spawn 相关段落）。


## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [FEP Agent Hub](./s2mon123-fep-agent-hub.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/v0v1kkk-freecad-engineering.md)
- [V0v1kkk/freecad-engineering 官方仓库](<https://github.com/V0v1kkk/freecad-engineering>)
- 源码核对（2026-11-02）：V0v1kkk/freecad-engineering@6d48362，读过 README.md、plugin.json、mcp.json、skills/freecad-engineering/SKILL.md、skills/freecad-engineering-bootstrap/SKILL.md 与 scripts/bootstrap_probe.py、scripts/render_mcp_config.py、skills/cad-viewer/SKILL.md、skills/cad-viewer/scripts/viewer/package.json、skills/cad-viewer/scripts/viewer/backend/server.mjs；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [项目 README](<https://github.com/V0v1kkk/freecad-engineering/blob/main/README.md>)