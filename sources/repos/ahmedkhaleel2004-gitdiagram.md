# GitDiagram（ahmedkhaleel2004/gitdiagram）

> 来源归档（ingest）

- **类型：** repo / AI codebase visualization / MCP
- **主仓库：** <https://github.com/ahmedkhaleel2004/gitdiagram>
- **在线入口：** <https://gitdiagram.com/> — [站点归档](../sites/gitdiagram-com.md)
- **许可证：** MIT
- **版本快照：** package.json 标记 0.1.0（项目版本字段）；入库日核对
- **入库日期：** 2026-10-08
- **一句话说明：** 通过 GitHub API 读取仓库树和限量源码片段，让模型提出架构图与简要说明，再校验路径和图结构、编译为 Mermaid，提供交互浏览、代码跳转、导出及 MCP 访问。
- **沉淀到 wiki：** 是 → [GitDiagram 实体页](../../wiki/entities/gitdiagram.md)

## 仓库要点

| 面向 | 归纳 |
|------|------|
| 输入 | GitHub 仓库；支持公开仓库，网站也提供私有仓库令牌入口 |
| 分析 | 默认分支、递归文件树、README 与有界源码摘录；大仓库对模型上下文有上限 |
| 图生成 | 模型产出架构概览与图结构；服务端检查标识符、连通性、限制及文件路径，再确定性编译 Mermaid |
| 浏览与导出 | 组件链接回 GitHub 文件/目录；支持 PNG 导出、复制 Mermaid；图结果可持久化复用 |
| Agent 接口 | 远程 MCP endpoint：<https://gitdiagram.com/mcp>；可读公开仓库架构说明、组件/关系、Mermaid 与视频信息 |
| 讲解视频 | 约一分钟旁白视频；截至入库日新视频生成功能处于 early access，已有视频可观看 |
| 本地开发 | Bun、Cloudflare R2、Upstash Redis、OpenAI 或 OpenRouter 密钥；Next.js、React、TypeScript、Mermaid |

## 架构文档指出的边界

生产文档将 Vercel 列为实时应用运行环境，Cloudflare R2 用于图工件，Upstash Redis 用于额度/取消等协调，OpenAI 或 OpenRouter 负责生成。仓库也保留 Railway/Docker 灾备配方，但文档明确它不是当前常驻线上运行时。浏览器会对 Mermaid 源码及 SVG 做净化并限制 GitHub 链接。

## 使用边界

模型生成的架构图是对有限仓库证据的解释，不等同于编译器级依赖图、完整调用图或运行时观测。大型仓库使用有界摘录；读取私有仓库需令牌，使用托管站点或 MCP 前应自行审查当前代码处理与保留政策。

## 主要入口

- README 与安装说明：<https://github.com/ahmedkhaleel2004/gitdiagram>
- 架构说明：<https://github.com/ahmedkhaleel2004/gitdiagram/blob/main/docs/architecture.md>
- 本地开发：<https://github.com/ahmedkhaleel2004/gitdiagram/blob/main/docs/dev-setup.md>
- MIT 许可证：<https://github.com/ahmedkhaleel2004/gitdiagram/blob/main/LICENSE>

## Wiki 映射

- [GitDiagram 实体页](../../wiki/entities/gitdiagram.md)
- [Mermaid.js 实体页](../../wiki/entities/mermaid-js.md)