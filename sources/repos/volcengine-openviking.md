# OpenViking（volcengine/OpenViking）

- **标题：** OpenViking
- **类型：** repo
- **来源：** 火山引擎 Viking 团队（字节跳动）
- **链接：** <https://github.com/volcengine/OpenViking>
- **项目页：** <https://openviking.ai/>
- **文档：** <https://docs.openviking.ai/>
- **许可：** AGPL-3.0
- **入库日期：** 2026-10-15
- **一句话说明：** 面向 AI Agent 的开源上下文数据库，将记忆、知识资源和技能统一组织为可浏览、可检索的虚拟文件系统。
- **沉淀到 wiki：** [wiki/entities/openviking-context-database.md](../../wiki/entities/openviking-context-database.md)

## 开源状态核查（2026-10-15）

- **已开源：** 官方 GitHub 仓库公开，提供 Python 实现、文档和示例，许可证为 AGPL-3.0。
- 官方项目页与 README 均链接到该仓库和文档站；本项目是上下文存储与检索软件，不是机器人策略、模型权重或机器人数据集。

## 核心设计（官方 README）

- 用 viking:// URI 将资源、用户记忆、技能和会话组织成目录树。
- 提供 ls、tree、read、write、grep 等文件式浏览操作；目录可带有生成摘要。
- 采用 L0 摘要、L1 概览、L2 原始详情的渐进式上下文读取，并支持按目录范围执行检索。
- 会话归档可提取可检查、可编辑的记忆；VikingBot 可进一步将资料编译为 wiki、知识图谱或报告。

## 对 wiki 的映射

- [OpenViking 上下文数据库](../../wiki/entities/openviking-context-database.md)
