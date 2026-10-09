# morluto/rea

> 来源归档（ingest）

- **标题：** REA（Reverse Engineer Anything）
- **类型：** repo / AI agent / MCP / CLI / 软件逆向工程
- **代码：** <https://github.com/morluto/rea>（**已开源**，MIT）
- **官方项目页：** <https://rea.tools/>
- **分发包：** <https://www.npmjs.com/package/rea-agents>
- **文档：** <https://rea.tools/guides/>
- **入库日期：** 2026-10-09
- **一句话说明：** 把本机反汇编、程序结构分析和受控运行观测等能力封装为统一 MCP Server 与 CLI，让编码代理可以分步调查二进制、JavaScript/Electron、网站、.NET、APK、固件等目标，并获得带来源证据的结果。

## 官方仓库核查

GitHub 仓库为公开 MIT 项目；README 链接至官方站点、npm 包与使用指南。项目自身是工具编排/接口层，不是反编译器：深度原生二进制分析依赖用户配置 Hopper、Ghidra 或 IDA；不同目标还需要浏览器、JADX/JDK、Binwalk/Unblob、pwntools 等相应工具。README 声明目标分析在本机进行，但 MCP 结果会返回给所连接的 Agent/model provider；运行时采集会以当前用户权限运行或交互目标。

## 资料映射

- 官方站点归档：[REA 项目页](../sites/rea-tools.md)
- 主实体：[REA：面向编码代理的软件逆向调查工具](../../wiki/entities/rea.md)
- 协议背景：[Model Context Protocol](../../wiki/concepts/model-context-protocol.md)
- 邻近工具：[Agent Reach](../../wiki/entities/agent-reach.md)

## 官方入口

- <https://github.com/morluto/rea>
- <https://rea.tools/>
- <https://www.npmjs.com/package/rea-agents>
- <https://rea.tools/guides/>
