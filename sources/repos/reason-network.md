# Reason Network

> 来源归档

- **标题：** Reason Network
- **类型：** repo
- **来源：** Reason Machines
- **链接：** https://github.com/reason-machines/reason-network
- **产品页：** https://reasonmachines.com/network
- **入库日期：** 2026-10-20
- **一句话说明：** 面向多种编码代理的跨设备协作通信引擎：发现团队成员正在运行的代理会话，并在代理之间转发消息、协作请求与用户审批。
- **代码：** https://github.com/reason-machines/reason-network（**已开源**；MIT OR Apache-2.0 双许可证）
- **开源边界：** 开源的是负责设备间通信的引擎；Reason CLI 负责登录、安装和配置，官方说明其免费但不开源。
- **沉淀到 wiki：** 暂不新建实体页；作为 AI 编码代理协作工具归档，不属于机器人运行时框架。

---

## 内容范围

仓库 README 将 Reason Network 描述为运行在各台电脑上的轻量后台服务。它能展示团队代理会话的成员、仓库、分支、忙闲状态和当前任务，并支持代理间消息发送、收件箱检查以及需要用户确认的协作请求。官方列出的接入对象包括 Claude Code、Codex、Cursor、Gemini CLI、Copilot CLI、Grok CLI、opencode、Pi、DeepSeek Harness 和 Reason。

## 通信与安全

设备间优先建立点对点连接；直连失败时可经中继转发加密流量。离线消息会等待对方重新上线后发送。官方说明设备间流量端到端加密，网络成员通过邀请加入；具体设计见仓库的 `DESIGN.md`。

README 列出的 MCP 工具包括 `list_team`、`send_message`、`ask_user`，以及消息无法主动推送时使用的 `check_inbox`。该项目用于开发代理协同，不控制机器人硬件，也不提供机器人通信总线或运动控制接口。

## 参考来源

- [Reason Network 官方 GitHub 仓库](https://github.com/reason-machines/reason-network)
- [Reason Network 产品页与 FAQ](https://reasonmachines.com/network)
- [仓库设计说明](https://github.com/reason-machines/reason-network/blob/main/DESIGN.md)
