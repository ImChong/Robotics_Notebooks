# Linger（simplexai-labs/linger）官方源码与技术文档

- **类型：** repo / 本地个人智能体平台
- **标题：** Linger — An Open-Source Personal Agent That Runs Fully on Your PC
- **代码：** https://github.com/simplexai-labs/linger
- **官网：** https://heylinger.com/
- **技术文档：** https://docs.heylinger.com/
- **核对版本：** 上游 main 提交 [cd0e23af6cacbca311bfe1964e11ea6c73cc7562](https://github.com/simplexai-labs/linger/commit/cd0e23af6cacbca311bfe1964e11ea6c73cc7562)（2026-11-19 UTC）。
- **核对范围：** 官方 README、docs/architecture.md、docs/local-inference.md、docs/wiki-and-memory.md、docs/proactive.md、docs/task-delegation.md、docs/desktop-and-browser.md、docs/team-md.md、docs/self-hosting.md 及 LICENSE；核对主要架构、运行入口、使用边界与授权条款。不是逐文件源码审计，也未在本机实测。
- **通读：** 已完成
- **收录日期：** 2026-11-20
- **源代码开放状态：** 已公开主仓与本地/自托管实现；授权是修改版 Apache 2.0，不等同无附加限制的 Apache-2.0。

## 一手资料与入口

- [README](https://github.com/simplexai-labs/linger/blob/cd0e23af6cacbca311bfe1964e11ea6c73cc7562/README.md)：定位、安装方式、硬件分档与产品功能。
- [Architecture](https://github.com/simplexai-labs/linger/blob/cd0e23af6cacbca311bfe1964e11ea6c73cc7562/docs/architecture.md)：Rooms / Chat → Computer Daemon → Runtime → LLM Gateway；工具调用经 Guardrail。
- [Local Inference](https://github.com/simplexai-labs/linger/blob/cd0e23af6cacbca311bfe1964e11ea6c73cc7562/docs/local-inference.md)：Local model / Subscription / BYOK 三条模型路径；本机代理与后台记忆模型归属。
- [Wiki and Memory](https://github.com/simplexai-labs/linger/blob/cd0e23af6cacbca311bfe1964e11ea6c73cc7562/docs/wiki-and-memory.md)：结构化个人 wiki、原始资料搜索、会话记忆提取与整合。
- [Proactive](https://github.com/simplexai-labs/linger/blob/cd0e23af6cacbca311bfe1964e11ea6c73cc7562/docs/proactive.md)：Another Me 主动跟进、安静时段与反馈规则。
- [Task Delegation](https://github.com/simplexai-labs/linger/blob/cd0e23af6cacbca311bfe1964e11ea6c73cc7562/docs/task-delegation.md)：Meta/AlterEgo 与任务状态机。
- [Desktop and Browser](https://github.com/simplexai-labs/linger/blob/cd0e23af6cacbca311bfe1964e11ea6c73cc7562/docs/desktop-and-browser.md)：桌面与浏览器复用范围。
- [Self-hosting](https://github.com/simplexai-labs/linger/blob/cd0e23af6cacbca311bfe1964e11ea6c73cc7562/docs/self-hosting.md)：Docker Compose、账号创建、Computer Daemon 配置。
- [LICENSE](https://github.com/simplexai-labs/linger/blob/cd0e23af6cacbca311bfe1964e11ea6c73cc7562/LICENSE)：修改版 Apache 2.0 附加条款。

## 核心核对结果

1. **个人电脑执行：** Computer Daemon 从 Rooms/Chat 接收任务，驱动 Claude Code、Codex 或 Pi Runtime；模型经 Local Gateway 接本地推理或用户选用的远程提供方；工作目录、记忆/个人 wiki 位于用户设备（部署方式仍影响安全边界）。
2. **知识组织：** 将邮件、会议和聊天整理为 Client / Contact / Project / Topic / Session 类型化 Markdown 页面；通过 `teamhub knowledge` 搜索 wiki 与原始信息；代理个人记忆与跨代理共用的个人 Wiki 分层。
3. **主动式代理：** Another Me 可从邮箱/日历等事件形成提醒与跟进；反馈规则迭代。任务可委托 AlterEgo，包含人工审批/停止机制。
4. **计算机工具：** 文档描述 Guardrail → Cua / Playwright / Shell；Linux 桌面行为与 Mac/Windows 不同，跨系统不能假设权限、登录态和浏览器复用完全一致。
5. **启动路径：** README 提供 macOS（Apple Silicon）、Windows x64、Linux x64 桌面发行包路径；`docs/self-hosting.md` 给出的可执行自托管入口是 `docker compose up --build --detach --wait`，然后 `teamhub-account create-owner`，浏览器访问 localhost:3000，连接 Computer Daemon/Runtime。
6. **许可：** LICENSE 允许内部商业使用，但对向组织外部第三方提供托管服务、以及修改/移除前端 LOGO 与版权声明设有商业许可要求；第三方组件以 NOTICE 为准。

## 不能直接照单全收的宣传数字

- README、官网列出的模型速度、上下文、智力分数和竞品对标是**项目自述/特定硬件与测试条件**，本次未做独立复测，不作为通用性能承诺。
- “数据留在本地”有条件：一旦启用云模型、外部连接器、第三方语音识别/实时语音，仍会按所选服务向外发送必要数据。
- “开源”不等于 Apache-2.0 原版或可无限制托管商用；实际约束以 LICENSE 原文为准。

## 对 wiki 的映射

- [Linger 独立项目实体](../../wiki/entities/linger-simplexai.md)
- [相关：Hermes Agent](../../wiki/entities/hermes-agent.md)
- [相关：LLM Wiki](../../wiki/concepts/llm-wiki-karpathy.md)
