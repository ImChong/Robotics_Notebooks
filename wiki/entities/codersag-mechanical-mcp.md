---
type: entity
project_id: codersag-mechanical-mcp
code: "https://github.com/codersag/mechanical-mcp"
tags:
  - engineering-tools
  - caa-cfd
  - mcp
status: complete
updated: 2026-10-07
summary: "Mechanical MCP（PyMechanical gRPC）（codersag/mechanical-mcp）是通过 PyMechanical gRPC 连接 ANSYS Mechanical，工具覆盖几何/材料、网格、边界条件、求解与结果、报告及 IronPython 脚本"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/codersag-mechanical-mcp.md
---

# Mechanical MCP（PyMechanical gRPC）（codersag/mechanical-mcp）

## 一句话定义

Mechanical MCP（PyMechanical gRPC）（codersag/mechanical-mcp）是通过 PyMechanical gRPC 连接 ANSYS Mechanical，工具覆盖几何/材料、网格、边界条件、求解与结果、报告及 IronPython 脚本。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MCP | Model Context Protocol | 代理与工程软件工具的交互协议 |
| CAE | Computer-Aided Engineering | 计算机辅助工程软件与分析 |
| API | Application Programming Interface | 软件可供自动化调用的接口 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。通过 PyMechanical gRPC 连接 ANSYS Mechanical，工具覆盖几何/材料、网格、边界条件、求解与结果、报告及 IronPython 脚本。

## 核心原理

**项目自身范围：** MCP server connecting AI clients to ANSYS Mechanical through PyMechanical gRPC.

**工作流：** 用户启动或定位 Mechanical gRPC 端口，MCP 客户端调用服务器连接会话，执行建模、求解和结果提取。


**仓库映射说明：** 文章未给出该项目完整的 owner/repository URL；此处按文章中的项目名/描述与公开 GitHub 仓库匹配。引用时请以本页链接的仓库 README 为准。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：通过 PyMechanical gRPC 连接 ANSYS Mechanical，工具覆盖几何/材料、网格、边界条件、求解与结果、报告及 IronPython 脚本。

## 局限与风险

依赖 ANSYS Mechanical 2023 R1+ 与授权；脚本执行能力需在受信任环境内使用，不能省略结构结果审查。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 Apache-2.0。 涉及商业软件时，安装包和有效许可证需由使用者自行提供。

## 源码运行时序图

入口：`pip install -e .` 后，MCP 客户端以 `"command": "ansys-mechanical-mcp"` 启动服务（`pyproject.toml` 的 `[project.scripts]` 指向 `mechanical_mcp_server:main`，经 `mcp.run()` 走 stdio）；Mechanical 端需在 Workbench 中 `StartGrpcServer()` 取得端口，再调用 `connect_to_mechanical(port)`。

```mermaid
sequenceDiagram
    autonumber
    participant C as MCP 客户端
    participant SRV as mechanical_mcp_server.py FastMCP
    participant TW as _tool 包装 线程池
    participant CI as _connect_impl 连接与握手
    participant RUN as _run / _execute
    participant MS as mechanical_scripts.py 脚本构造
    participant MECH as Mechanical 2026 R1 gRPC 会话

    C->>SRV: 启动 ansys-mechanical-mcp（mcp.run，stdio）
    C->>SRV: tools/call connect_to_mechanical(port)
    SRV->>TW: run_in_executor，guarded(port)
    TW->>TW: ms.integer 校验端口，检查后台求解是否占用
    TW->>CI: _connect(port) → _connect_impl(port)
    CI->>CI: _port_open 探测 127.0.0.1，_silent 探测是否说 gRPC
    CI->>MECH: mech.connect_to_mechanical(port, connect_timeout)
    CI->>RUN: _execute(conn, _INFO_SCRIPT) 握手检查
    RUN->>MECH: run_python_script(_wrapper(脚本文件, 输出文件))
    MECH-->>RUN: 输出 project 与 analyses 的 JSON
    CI-->>TW: 握手通过后才替换当前连接
    TW-->>C: {"ok": true, port, project, analyses}
    C->>SRV: tools/call assign_material(body_name, material_name)
    SRV->>TW: _tool 包装，线程池执行 _scripted
    TW->>MS: assign_material_script(body_name, material_name) 生成 IronPython 脚本
    TW->>RUN: _checked_run → _check_connection
    RUN->>RUN: _acquire_mech 获取 _MECH_LOCK
    RUN->>MECH: run_python_script(包装后的脚本)
    MECH-->>RUN: 输出文件中的打印结果
    RUN-->>TW: raw 结果，_result 解析 ok 字段
    TW-->>C: paths.redact 后的 JSON
    alt gRPC 调用在传输中失败
        RUN->>RUN: _close_channel，不重发脚本
        RUN-->>C: Error，提示先检查模型再重试
    end
    Note over SRV,MECH: run_mechanical_script 默认禁用，需 MECHANICAL_MCP_ENABLE_SCRIPT=1
```

图依据：codersag/mechanical-mcp@c247356，实际读过 README.md、pyproject.toml（[project.scripts]）、mechanical_mcp_server.py（_tool、_connect_impl、_connect、_check_connection、_acquire_mech、_run、_execute、_checked_run、_scripted、connect_to_mechanical、assign_material、run_mechanical_script、main 段）；mechanical_scripts.py 与 mcp_paths.py 未打开，脚本构造与路径脱敏只按调用名标注。

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [ANSYS Workbench MCP](./hongwenwang36-eng-ansys-workbench-mcp.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/codersag-mechanical-mcp.md)
- [codersag/mechanical-mcp 官方仓库](<https://github.com/codersag/mechanical-mcp>)

- 源码核对（2026-11-02）：codersag/mechanical-mcp@c247356，读过 README.md、pyproject.toml、mechanical_mcp_server.py（连接、执行、工具包装与 main 段）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [项目 README](<https://github.com/codersag/mechanical-mcp/blob/main/README.md>)