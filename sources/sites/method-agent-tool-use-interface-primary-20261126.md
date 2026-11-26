# Model Context Protocol：Tools（2025-06-18）

- **URL：** <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>
- **类型：** 官方文档或作者工程文章
- **核验日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** Capabilities、Listing/Calling、Tool 与 Tool Result 数据类型、Error Handling、Security Considerations。

## 来源摘要

该固定版本规定工具发现和调用消息。工具声明输入 schema，可发布结果 schema；协议错误与工具执行错误经不同通道报告。服务端校验工具输入并实施访问控制，客户端对敏感操作呈现确认并检查输出，本页不推断别的协议修订或客户端也必须用相同界面。

关联：[方法页](../../wiki/methods/agent-tool-use-interface.md)。
