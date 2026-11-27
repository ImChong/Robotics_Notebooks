# OpenPE v0.1.0 Beta · 源码核对

- **类型：** repo
- **原文：** https://github.com/Open-Power-Electronics/OpenPE
- **项目页：** https://open-power-electronics.github.io/
- **固定版本：** [v0.1.0 / 2c5d3b294d7fe45ea12cfd1fc7380f353ff5df3d](https://github.com/Open-Power-Electronics/OpenPE/tree/2c5d3b294d7fe45ea12cfd1fc7380f353ff5df3d)
- **发布日期：** 2026-11-23
- **许可证：** Apache-2.0；示例和第三方资产另有署名与条款。
- **核对日期：** 2026-11-27
- **核对范围：** Beta 发布页、README 和文档索引；架构分层、API 的能力发现与 Agent 执行契约、插件范围、布局预览应用与 STEP 导出契约、EMC Flyback 例程说明、SPICE 与温度模型相关章节；追踪 Agent Review/Auto 适配器和布局事务源码。非全仓逐文件审计。
- **通读：** 已完成

| 入口 | 技术证据 |
|---|---|
| [Beta 发布](https://github.com/Open-Power-Electronics/OpenPE/releases/tag/v0.1.0) | Windows x64、macOS arm64 附安装包和 SHA-256；未签名，macOS 未公证。Linux x64 未发布安装包和求解器下载，需源码运行并本地构建求解器。 |
| [架构](https://github.com/Open-Power-Electronics/OpenPE/blob/v0.1.0/docs/ARCHITECTURE.md)与[Agent API](https://github.com/Open-Power-Electronics/OpenPE/blob/v0.1.0/docs/API.md#agent-tools) | GUI 与受限 Agent 接入共享 Uniform API；Core 维护工程事务。对象级和操作级是命令抽象层，任务级尚无注册；这不是三层权限。 |
| [Review/Auto 源码](https://github.com/Open-Power-Electronics/OpenPE/blob/v0.1.0/lib/chat-stream.ts#L75)与[布局事务](https://github.com/Open-Power-Electronics/OpenPE/blob/v0.1.0/core/layout-commands.ts#L125) | Review 阶段将需确认操作返回为待处理，不等于写入；布局应用会重新计算建议、比对位姿并以一次可撤销事务提交。 |
| [EMC Flyback 校核](https://github.com/Open-Power-Electronics/OpenPE/blob/v0.1.0/docs/EMC-FLYBACK.md) | 内置示例源自 AI-PCB-layout 的 CC BY 4.0 设计。ngspice 47 跑 10 ms、最大步长 5 ns；本地示例载荷与上游参考不同，电气、温度与外壳检查分开。 |
| [插件文档](https://github.com/Open-Power-Electronics/OpenPE/blob/v0.1.0/docs/PLUGINS.md)与[热模型](https://github.com/Open-Power-Electronics/OpenPE/blob/v0.1.0/docs/THERMAL.md) | 内置五项，另支持限定输入与项目读取权限的 QuickJS 扩展。板温不等于器件结温，EMC 几何审查不等于认证。 |

本次额外用 Node 直接调用固定版本的 Review/Auto 适配器和 Agent bridge，以模拟工作台通过 7 条断言，核查待审命令未执行、Auto 进入执行、读取响应及取消后的状态。未打开桌面客户端、连接供应商、运行求解器或检验实物。
