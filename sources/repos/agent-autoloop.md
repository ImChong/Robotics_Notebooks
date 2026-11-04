# Agent Autoloop

- **类型：** repo
- **链接：** <https://github.com/KKKKhazix/agent-autoloop>
- **许可：** MIT
- **核查版本：** `cecec75be3ca26e6ca1d6833b50152ee9450088d`，2026-11-04
- **开源状态：** 已开源，含 TypeScript 内核、运行外壳、worker、插件、测试与示例。
- **入口：** `bin/agent-autoloop.ts` → `cli/main.ts`；`agent-autoloop up` → `host/up.ts`。
- **资料：** [README](https://github.com/KKKKhazix/agent-autoloop/blob/cecec75be3ca26e6ca1d6833b50152ee9450088d/README.md)、[协议](https://github.com/KKKKhazix/agent-autoloop/blob/cecec75be3ca26e6ca1d6833b50152ee9450088d/docs/protocol.md)。
- **知识页：** [Agent Autoloop](../../wiki/entities/agent-autoloop.md)。

仓库描述了本地运行的事件驱动 Agent 调度：来源与定时事件按规则派发，worker 启动 CLI Agent，解析末条回复中的回执；需要人工判断时以卡片暂停事项，回答产生后续工作。README 给出安装条件和实测范围，协议解释幂等、重试、权限与局限。

**核查范围与待办：** 已阅读 README、部分协议及主要入口与 worker 回执路径；协议全文尚未逐节复核，未实机运行项目。待完成全文复核及独立运行验证。
