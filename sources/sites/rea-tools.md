# REA（rea.tools 官方项目页）

> 来源归档

- **标题：** REA — Reverse Engineer Anything
- **类型：** site / 官方项目页与实践指南
- **来源：** REA 项目
- **链接：** <https://rea.tools/>
- **代码：** <https://github.com/morluto/rea>（项目页 Source 入口；MIT，已开源）
- **安装：** <https://www.npmjs.com/package/rea-agents>
- **指南：** <https://rea.tools/guides/>
- **入库日期：** 2026-10-09
- **一句话说明：** 编码代理借助 REA 检查软件如何工作：项目页通过 Windows 计算器、Chrome 恐龙游戏和 Notes 示例展示从目标代码/行为、证据解释到重建小型功能的调查链路。
- **开源状态：** 已开源；官方项目页链接 GitHub 源码。站点示例是工作流说明，不代表每类目标都能无依赖分析。

## 项目页核查（2026-10-09）

REA 的价值是把逆向分析器和调查工作流接到 Agent，而非承诺自动从二进制还原原始源码。官网首页提供 setup 指令 npx rea-agents@latest setup，要求查看并批准配置计划、重启 Agent。官方 showcase 展示了浏览器中的 JavaScript 分析、Windows Calculator 原生代码路径和跨 renderer/preload/main 的 Electron 跟踪。

分析运行于本机；但 Agent 接收工具返回的信息，模型数据处理仍受所用 provider 的政策影响。对于会启动目标或采集交互行为的功能，应先阅读对应指南并确认目标授权与运行副作用。

## 对 wiki 的映射

- 主实体：[REA](../../wiki/entities/rea.md)
- 源码归档：[morluto/rea](../repos/morluto-rea.md)
- 协议背景：[Model Context Protocol](../../wiki/concepts/model-context-protocol.md)

## 官方链接

- <https://rea.tools/>
- <https://rea.tools/guides/>
- <https://github.com/morluto/rea>
- <https://www.npmjs.com/package/rea-agents>
