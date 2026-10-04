# Mermaid.js（mermaid-js/mermaid）

> 来源归档（ingest）

- **类型：** repo / diagram-as-code / JavaScript / visualization
- **主仓库：** <https://github.com/mermaid-js/mermaid>
- **官方文档入口：** <https://mermaid.ai/open-source/>（旧文档域 <https://mermaid.js.org/>）
- **在线编辑器：** <https://mermaid.live/>
- **编辑器源码：** <https://github.com/mermaid-js/mermaid-live-editor>
- **命令行工具：** <https://github.com/mermaid-js/mermaid-cli>
- **许可证：** MIT（上述三个 GitHub 仓库的元数据均标注 MIT）
- **入库日期：** 2026-10-04
- **一句话说明：** Mermaid 将 Markdown 风格的文本定义解析并渲染为图表；主库负责语法与 SVG 渲染，Live Editor 用于即时预览，CLI 用于脚本化导出。

## 项目关系

| 组件 | 职责 | 官方入口 |
|------|------|----------|
| Mermaid 主库 | 定义图表语法、解析并渲染图表 | [GitHub](https://github.com/mermaid-js/mermaid) |
| Live Editor | 浏览器内编辑、预览、分享图表 | [网页](https://mermaid.live/) · [源码](https://github.com/mermaid-js/mermaid-live-editor) |
| Mermaid CLI | 在命令行或自动化脚本中生成图表文件 | [GitHub](https://github.com/mermaid-js/mermaid-cli) |
| 官方文档 | 入门、图类型语法、配置和集成说明 | [Open Source](https://mermaid.ai/open-source/) |

## 官方文档入口

- [Getting Started](https://mermaid.ai/open-source/intro/getting-started.html)
- [Flowchart 语法](https://mermaid.ai/open-source/syntax/flowchart.html)
- [Sequence Diagram 语法](https://mermaid.ai/open-source/syntax/sequenceDiagram.html)
- [Usage / API](https://mermaid.ai/open-source/config/usage.html)
- [集成清单](https://mermaid.ai/open-source/ecosystem/integrations-community.html)
- [CLI 文档](https://mermaid.ai/open-source/config/mermaidCLI.html)

## 开源状态与使用边界

主库、Live Editor 和 CLI 均在 mermaid-js GitHub 组织下公开维护，仓库元数据标记 MIT。主库可嵌入网页或文档工具；Live Editor 适合快速试写；CLI 适合批量导出。图语法是否能直接显示，仍取决于宿主平台集成的 Mermaid 版本、启用图类型和安全配置，应以目标环境实测为准。

## Wiki 映射

- [Mermaid.js 项目节点](../../wiki/entities/mermaid-js.md)
- [Diagram Design](../../wiki/entities/diagram-design.md)
- [Archify](../../wiki/entities/archify.md)
