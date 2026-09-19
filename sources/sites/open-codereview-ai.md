# open-codereview.ai（Open Code Review 官方站）

- **标题：** Open Code Review — AI-powered code review CLI
- **类型：** site / project-page
- **URL：** <https://open-codereview.ai>（文档：<https://open-codereview.ai/docs>）
- **入库日期：** 2026-09-19
- **代码：** <https://github.com/alibaba/open-code-review>（npm：`@alibaba-group/open-code-review`；仓库归档 [`sources/repos/open-code-review.md`](../repos/open-code-review.md)）
- **开源状态：** **已开源**（Apache-2.0；CLI、规则集、插件与 CI 集成文档均在 GitHub 公开）

## 一句话摘要

阿里巴巴从内部 AI 代码评审助手孵化出的 **Open Code Review（OCR）** 官方站点：强调 **确定性工程管线 × LLM Agent** 混合架构，提供 CLI、多编码代理插件、Delegation Mode 与 CI/CD 集成说明。

## 公开信息要点（截至入库日 2026-09-19）

- **产品定位：** 面向 Git diff / 全文件扫描的 **行级精确** AI 代码评审；内置多语言规则（NPE、线程安全、XSS、SQL 注入等）；兼容 OpenAI / Anthropic 等 provider。
- **与通用 Agent 的差异（站点 + README 一致口径）：** 确定性步骤保证 **文件选择、分包、规则匹配、评论定位/反思** 不被模型「偷懒」跳过；Agent 侧聚焦 **场景化 prompt 与 toolset**。
- **基准：** [AACR-Bench](https://huggingface.co/datasets/Alibaba-Aone/aacr-bench)（50 仓库 × 200 PR × 10 语言；80+ 高级工程师交叉标注）；README 称相较同模型通用 Agent **Precision / F1 更高、token 约 1/9**，Recall 有意偏低以减少噪声。
- **集成入口：** Claude Code / Codex / Cursor / Kimi Code / OpenCode 等插件；MCP Server；GitHub Actions / GitLab CI / Gerrit 文档。
- **Delegation Mode：** OCR 负责文件选择与规则解析，由宿主编码代理自行执行评审（无需 OCR 侧 LLM API key）。

## 为何值得保留

- **非 README 证据：** 安装、配置、Review Rules、Session Viewer、Telemetry 等结构化文档便于与 wiki 流程图对齐。
- **开源入口核验：** 项目页与 GitHub 互指；npm 包与 Release 二进制可独立安装，**不是**「宣称将开源」状态。

## 关联资料

- 代码归档：[`sources/repos/open-code-review.md`](../repos/open-code-review.md)
- wiki：[`wiki/entities/open-code-review.md`](../../wiki/entities/open-code-review.md)
