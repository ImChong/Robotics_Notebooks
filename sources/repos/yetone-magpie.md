# Magpie（yetone/magpie）— 多 Agent 本地模型网关

> 一手资料归档（ingest；2026-11-18）

- **标题：** magpie — Every agent's model. One place.
- **作者 / 组织：** yetone / Magpie 开源项目
- **类型：** repo（Go CLI / 桌面客户端 / 本地模型网关）
- **代码：** https://github.com/yetone/magpie
- **项目页：** https://usemagpie.ai/
- **原文：** https://github.com/yetone/magpie/blob/main/README.md
- **中文原文：** https://github.com/yetone/magpie/blob/main/README.zh-CN.md
- **架构原文：** https://github.com/yetone/magpie/tree/main/docs/subsystems
- **许可：** MIT（官方仓库 LICENSE / GitHub 元数据）
- **入库日期：** 2026-11-18
- **核对范围：** 通读 2026-11-18 官方英文 README 的产品定位、安装、Agent/Provider、Gateway、Routing、Privacy、CLI、部署、限制和许可章节；对照中文 README 的术语；阅读 docs/subsystems/README.md 与 agent-wiring.md 的入口/配置说明、gateway-routing.md 的总体职责与路由边界，以及源码 main.go、internal/cli/start_cli.go、internal/gateway/gateway_hooks.go、internal/provider/provider_order.go。大型 gateway.go、routing.go 及 70 万字符路由参考未逐行审计；未进行本地安装与联网实测。
- **通读：** 已完成
- **一句话说明：** 通过配置适配器和本地多协议 Gateway 为 Claude Code、Codex、Gemini CLI 等共享 API Key、订阅账号、模型路由、流式协议转换与故障转移。
- **沉淀到 wiki：** [Magpie](../../wiki/entities/magpie.md)
- **关联站点归档：** [官网](../sites/usemagpie-ai.md)

## 官方实现要点（基于上述核对范围）

| 部分 | 核对结论 | 一手路径 |
|------|----------|----------|
| 命令入口 | Go `main.main` 调用 `cli.Main(version)`，构建时注册内置插件 | [main.go](https://github.com/yetone/magpie/blob/main/main.go) |
| Agent 配置 | `internal/agent` 描述各 Agent 配置文件字段；连接/断开时修改或还原字段 | [agent-wiring.md](https://github.com/yetone/magpie/blob/main/docs/subsystems/agent-wiring.md) |
| 启动适配 | 某些仅启动时可配置的 Agent 用 `magpie <agent> run` 注入环境和模型参数 | [start_cli.go](https://github.com/yetone/magpie/blob/main/internal/cli/start_cli.go) |
| Gateway | 默认 `127.0.0.1:3425`，支持 OpenAI Chat/Responses、Anthropic Messages、Gemini 协议入口和上游转换 | [README](https://github.com/yetone/magpie/blob/main/README.md) / [routing](https://github.com/yetone/magpie/blob/main/docs/subsystems/gateway-routing.md) |
| 路由及回退 | 选择 Provider、Key、Subscription 账号；上游回复字节尚未发送前才可故障转移 | [gateway-routing.md](https://github.com/yetone/magpie/blob/main/docs/subsystems/gateway-routing.md) |
| 隐私遮蔽 | Settings → Privacy 中可选，本地将敏感内容替换成占位符，流式响应和工具调用输出也做对应还原；默认关闭 | [README](https://github.com/yetone/magpie/blob/main/README.md) |
| Gateway 回调 | `gatewayResponse` 包装响应并收集成功/失败后通知运行插件 | [gateway_hooks.go](https://github.com/yetone/magpie/blob/main/internal/gateway/gateway_hooks.go) |

## 开源与可运行入口核查

- **状态：已开源。** 可公开查看 Go 主程序、`internal/agent`、`internal/provider`、`internal/gateway`、插件与文档；仓库声明 MIT。
- **官方运行入口：** 安装脚本 `curl -fsSL https://usemagpie.ai/install.sh | sh`；也可使用 `go install github.com/yetone/magpie@latest`、Docker 或官方二进制。CLI 示例：`magpie provider add deepseek <API_KEY>` → `magpie claude deepseek/<model-id>`，并在新的 Claude Code 会话生效。
- **开源边界：** 代理模型权重和第三方推理服务不随代码发布，账号可用性与访问权限依赖各提供商/协议；未实测跨厂商功能。
- **隐私边界：** 默认关闭遮蔽；本地 loopback 地址不等于所有场景都免认证，远程/LAN 部署需按官方 Gateway Key、TLS 等安全配置。

## 面向机器人研发的映射

此工具适合在多种 coding agent 之间统一管理代码分析、脚本修订、CI 修复等任务使用的模型后端；**不承担 ROS2 实时控制、运动策略输出、执行器硬实时安全链路**。可与 [Hermes Agent](../../wiki/entities/hermes-agent.md)、[DeepSeek Harness](../../wiki/entities/deepseek-harness.md) 对照（后两者是 Agent 宿主/运行时，而 Magpie 偏模型连接与请求面）。

## 参考原文

- [英文 README](https://github.com/yetone/magpie/blob/main/README.md)
- [Subsystem 设计说明](https://github.com/yetone/magpie/blob/main/docs/subsystems/README.md)
- [官网](https://usemagpie.ai/)
