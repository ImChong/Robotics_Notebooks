# Next AI Draw.io 演示站（next-ai-drawio.jiang.jp）

> 来源归档

- **标题：** Next AI Draw.io — AI-Powered Diagram Creation Tool
- **类型：** site / project-page
- **URL：** <https://next-ai-drawio.jiang.jp/>
- **代码：** <https://github.com/DayuanJiang/next-ai-draw-io> — [`sources/repos/next-ai-draw-io.md`](../repos/next-ai-draw-io.md)
- **作者：** DayuanJiang
- **入库日期：** 2026-09-21
- **一句话说明：** 官方在线演示：聊天式自然语言生成/编辑 draw.io 图，支持 BYOK、多模型与实时画布预览；配套 `@next-ai-drawio/mcp-server` 供 Cursor / Claude 等 MCP 客户端调用。

## 开源核查（步骤 2.5，截至 2026-09-21）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **是** — GitHub README 与演示站页脚/赞助区均指向 `github.com/DayuanJiang/next-ai-draw-io` |
| 可运行实现 | **有** — 在线 demo、Docker、桌面 Release（Win/macOS/Linux）、`npm run dev` 自托管 |
| MCP 包 | **有** — `@next-ai-drawio/mcp-server`（npm，`npx` 一键） |
| 数据 / 权重 | **无**（LLM 由用户/API 提供；演示站可 BYOK，密钥存浏览器 localStorage） |
| 综合判定 | **已开源**（Apache-2.0） |

## 页面要点（2026-09-21 抓取）

- Hero：**Chat, Draw, Visualize**；英文/中文/日文 README 分流。
- 演示能力：自然语言建图、上传图片/PDF/文本复刻、云架构（AWS/GCP/Azure）、动画连接器、版本历史与 AI reasoning 展示。
- **Bring Your Own API Key**：聊天面板 Settings 配置 provider + key，仅存本地浏览器。
- 赞助与默认模型：演示站曾由 ByteDance Doubao / Atlas Cloud 等赞助 API 额度；README 注明 demo 可用 glm-4.7 等（以站点当时配置为准）。
- 部署入口：Vercel / EdgeOne Pages / Cloudflare Workers 一键部署文档链出 GitHub `docs/en/`。

## 关联资料

- 仓库归档：[`sources/repos/next-ai-draw-io.md`](../repos/next-ai-draw-io.md)
- Wiki：[`wiki/entities/next-ai-draw-io.md`](../../wiki/entities/next-ai-draw-io.md)
