# Huawei HarmonyOS Developer Documentation Center（华为鸿蒙开发者文档中心）

> 官方文档来源归档；本条目保存核实后的入口、内容范围和版本边界，不转载完整文档。

- **标题：** 开发者文档中心 / HarmonyOS Developer Documentation Center
- **类型：** site（官方开发文档导航、AI 文档检索服务）
- **机构：** 华为开发者联盟（HUAWEI Developers）
- **官方网站：** <https://developer.huawei.com/consumer/cn/doc/doccenter>
- **MCP 官方指南：** <https://developer.huawei.com/consumer/cn/doc/start/hosknowledgemcp-0000002664603963>
- **llms.txt 官方指南：** <https://developer.huawei.com/consumer/cn/doc/start/llms-0000002670606713>
- **一级 llms.txt 索引：** <https://developer.huawei.com/config/llms.txt>
- **核对日期：** 2026-11-11
- **核对范围：** 逐段通读用户提供的文档中心主页、其直链「开发者知识 MCP」指南、直链「llms.txt 文档索引」指南及完整的一级 llms.txt 文本；核验官方公告所述新版文档中心 2026-09-28 全面开放。**未**通读其二级索引所链接的所有具体 API/设计/工具文档，也未调用远程 MCP 验证运行响应。
- **通读：** 已完成
- **一句话说明：** 面向 HarmonyOS 5.0 及以上版本的统一官方文档入口；网页分类导航外，还提供官方开发者知识 MCP 和供 LLM/RAG 使用的两级 llms.txt 索引。
- **沉淀到 wiki：** [华为鸿蒙开发者文档中心](../../wiki/entities/huawei-harmonyos-developer-doc-center.md)

## 已核实内容（官方一手资料）

### 文档中心的适用范围和导航

- 官方主页说明：文档主要适用于 **HarmonyOS 5.0 及以上**，覆盖版本说明、DevEco 工具、设计、指南与 API 参考、开发与测试、上架分发与运营、问题处理、最佳实践和行业解决方案。
- 人工入口：左侧「全部文档」分类、按目录标题过滤；入门顺序建议为应用开发导读 → DevEco Studio 环境搭建 → 首个 HarmonyOS 应用 → ArkTS。
- 旧版文档（HarmonyOS 3.1/4.0 及以下）由独立入口提供，不应将此中心当作所有历史版本的唯一来源。

### 开发者知识 MCP

- 官方 MCP 地址：`https://connect-api.cloud.huawei.com/api/developerknowledge/mcp`。
- `searchDocuments` 检索匹配片段与文档标识；`getDocumentsById` 按检索结果中的 `parent` 标识获取正文（官方说明一次最多 10 份）。
- 官方说明与站内文档准实时同步、发布后的内容同步延迟 **不超过 30 分钟**；这是平台发布的服务承诺，并非本次独立测得的时延。
- 官网给出远程 HTTP MCP 配置、DevEco Code 命令行及连接状态验证步骤；未据此断言所有客户端都已兼容。

### llms.txt

- 一级索引为 `https://developer.huawei.com/config/llms.txt`，其文件头标示 `update_time: 2026-09-17 00:00:00`（读取日：2026-11-11）。
- 两级结构：一级按开发者文档分类链接到二级 `docs-*.txt`；二级再链接到具体 Markdown 文档。可作为 AI 工具检索和私有知识库/RAG 同步的采集入口。
- 官方明确 `llms.txt` 是开放提案，**不是**所有 AI 工具都支持的强制标准。索引内容可能更新，不能把当前结构当成固定 API 契约。

## 开源与可运行性边界

| 对象 | 本次核查结论 |
|---|---|
| 文档中心网页及索引 | 公开可访问的官方资料 |
| 开发者知识 MCP | 官方提供远程服务接口及调用说明；**不等同于公开 MCP 服务端源码** |
| 示例代码库 | 导航页提供独立链接；本次未审计其仓库及许可 |
| 本文档中心的可运行官方源码 | 本次指定资料未提供可辨识的站点/MCP 服务端运行入口，源码时序图不适用 |

## 与机器人知识库的关系

该入口主要服务 HarmonyOS 应用与多设备开发，并非机器人控制栈或 ROS 2 的技术参考；对本库的价值是**官方端侧应用文档检索与 AI grounding 模式**，不应推导为已支持特定机器人硬件/实时控制。

## 对 wiki 的映射

- [华为鸿蒙开发者文档中心](../../wiki/entities/huawei-harmonyos-developer-doc-center.md)：文档导航、MCP/llms.txt 双路径与工程边界。
- [Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)：解释官方来源接入 AI 工具时仍需版本、出处与验证。
