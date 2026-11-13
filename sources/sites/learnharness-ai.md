# LearnHarness 官方站（learnharness.ai）

> 来源归档（ingest）

- **类型：** 网站 / 官方产品页
- **入口：** <https://learnharness.ai/>
- **代码：** <https://github.com/amosblomqvist/learnharness>（**已开源**，AGPL-3.0）
- **项目维护者：** Amos Blomqvist
- **收录与核验日期：** 2026-11-13
- **通读：** 已完成
- **核对范围：** 已通读官网 `I. The idea`、`II. Teaching`、`III. Practice`、`IV. Graph`、`V. The terms` 和下载区的文字说明，并对照上游 README 与 Getting Started 确认功能及限制。

## 官网内容定位

- **教学：** Probe → Plan → Teach。输入知识目标，先试探已知边界，构建最小依赖树，再通过 predict → explain → check 推进。
- **练习：** 与 Teach 分开的 Practice 阶段，在变式题中检验提取与迁移；提示按推理逐步调整。
- **知识图谱：** 由教学会话提取 proposition / reasoning chain，跨会话共享上下文，而非使用熟练度分数构建图谱。
- **应用形态：** 官网提供 macOS、Windows、Linux 下载；源码仓明确列出 macOS Apple Silicon、Windows/Linux x64 的具体安装包类型。
- **收费边界：** 应用本身无 LearnHarness 订阅费或用量加价；模型提供商、CLI、听写与搜索服务的资费依其配置决定。
- **隐私边界：** 学习记录存放本地，但通过外部模型、搜索和听写请求的内容可能发送给第三方。

## 开源状态及复核

- **结论：已开源**，官网 `Source` 链向 [官方 GitHub](https://github.com/amosblomqvist/learnharness)。
- **许可：** AGPL-3.0；官方说明系统提示词可检查与修改。
- **独立复现：** 本次未安装或运行桌面端，不能以产品演示宣称学习效果已实测验证。
- **代码固定核验版本：** [cb25c8e](https://github.com/amosblomqvist/learnharness/tree/cb25c8e49b85eed9e78315584731cad73a0118a6)。

## 对 wiki 的映射

- [LearnHarness 实体节点](../../wiki/entities/learnharness.md)
- [官方源码归档](../repos/learnharness.md)
