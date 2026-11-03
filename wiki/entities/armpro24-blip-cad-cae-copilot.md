---
type: entity
project_id: armpro24-blip-cad-cae-copilot
code: "https://github.com/armpro24-blip/cad-cae-copilot"
tags:
  - engineering-tools
  - caa-cfd
  - cad
status: complete
updated: 2026-10-07
summary: "CAD CAE Copilot（armpro24-blip/cad-cae-copilot）是以自然语言生成 CAD/CAE 任务，使用 build123d 与 OpenCASCADE 创建真实可编辑几何，包含参数、稳定拓扑引用、确定性检查和 MCP 工具"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/armpro24-blip-cad-cae-copilot.md
---

# CAD CAE Copilot（armpro24-blip/cad-cae-copilot）

## 一句话定义

CAD CAE Copilot（armpro24-blip/cad-cae-copilot）是以自然语言生成 CAD/CAE 任务，使用 build123d 与 OpenCASCADE 创建真实可编辑几何，包含参数、稳定拓扑引用、确定性检查和 MCP 工具。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CAD | Computer-Aided Design | 计算机辅助设计与几何建模 |
| MCP | Model Context Protocol | 代理调用 CAD/可视化工具的协议 |
| FEM | Finite Element Method | 结构有限元分析方法 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。以自然语言生成 CAD/CAE 任务，使用 build123d 与 OpenCASCADE 创建真实可编辑几何，包含参数、稳定拓扑引用、确定性检查和 MCP 工具。

## 核心原理

**项目自身范围：** AI-native CAD/CAE workbench for text-driven geometry and simulation tasks.

**工作流：** 将文本意图转换为参数化几何，再通过拓扑引用和规则检查约束模型；MCP 让代理调用建模与审查工具。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：以自然语言生成 CAD/CAE 任务，使用 build123d 与 OpenCASCADE 创建真实可编辑几何，包含参数、稳定拓扑引用、确定性检查和 MCP 工具。

## 局限与风险

自然语言建模的几何正确性仍需检查；文本提示不提供公差、制造可行性或结构安全保证。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 MIT。 

## 源码运行时序图

入口：`aieng-workbench-mcp`（即 `python -m aieng_workbench_mcp`，`--http` 为 SSE，默认 stdio；`--backend-url` 可选转发到 FastAPI 后端 `uvicorn app.main:app`）

```mermaid
sequenceDiagram
    participant Agent as MCP 客户端（Claude Code / Codex 等）
    participant Main as aieng_workbench_mcp.__main__ → app.mcp_server.main
    participant Srv as mcp_server._build_mcp_server / _make_handler
    participant Fwd as mcp_server._forward_to_backend
    participant API as routers/cad_live.py agent_invoke_tool<br/>POST /api/agent/invoke-tool
    participant RT as app.runtime.invoke_tool
    participant Gen as cad_generation.execute_build123d_code
    participant Sub as build123d runner 子进程

    Agent->>Main: 启动进程（stdio 或 --http）
    Main->>Srv: _build_mcp_server()
    Srv->>Srv: create_app() 填充工具注册表；list_tools_for_mcp()；FastMCP.add_tool
    Main->>Agent: mcp.run("stdio") 或 uvicorn.run(mcp.sse_app())
    Agent->>Srv: 调用工具（如 cad_execute_build123d，点号换成下划线）
    Srv->>Srv: _preflight（guide 守卫、检查模式拦截）
    alt requires_approval
        Srv->>Agent: 审批（broker / elicit 模式）
    end
    Srv->>Fwd: _execute：若配置 backend URL 则转发
    Fwd->>API: GET /api/health 探活后 POST invoke-tool
    alt 工具为 cad.execute_build123d
        API->>Gen: execute_build123d_code(settings, project_id, inp)
        Gen->>Sub: _execute_build123d_code_streaming（subprocess.Popen runner.py）
        Sub-->>Gen: STEP / STL / GLB / topology.json
        Gen->>Gen: _write_cad_artifacts 写入 .aieng 包；_publish_preview_to_viewer
    else 其他工具
        API->>RT: invoke_tool(tool, inp)
    end
    API-->>Fwd: 结果 JSON（并经 agent_activity 发布 tool_started/tool_completed 事件）
    Note over Srv,RT: 后端不可达时 _execute 回退为进程内 _rt.invoke_tool
    Srv-->>Agent: _finalize_result
```

图依据：armpro24-blip/cad-cae-copilot@b98b4d5，读过 `README.md`、`aieng-ui/backend/pyproject.toml`、`aieng_workbench_mcp/__main__.py`、`app/mcp_server.py`、`app/main.py`、`app/runtime.py`、`app/routers/cad_live.py`、`app/cad_generation.py`（部分段落）。

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [AnkusDrive](./gchen19-ankusdrive.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/armpro24-blip-cad-cae-copilot.md)
- [armpro24-blip/cad-cae-copilot 官方仓库](<https://github.com/armpro24-blip/cad-cae-copilot>)
- 源码核对（2026-11-02）：armpro24-blip/cad-cae-copilot@b98b4d5，读过 README.md、aieng-ui/backend/pyproject.toml、aieng_workbench_mcp/__main__.py、app/mcp_server.py、app/main.py、app/runtime.py、app/routers/cad_live.py、app/cad_generation.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [项目 README](<https://github.com/armpro24-blip/cad-cae-copilot/blob/main/README.md>)