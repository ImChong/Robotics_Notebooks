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
- **核对日期：** 2026-11-12（官方仓库默认分支与项目文档）
- **核对范围：** 已通读主仓 `README.md` 的 About、Features、Examples、Release、Related projects、Security and safe diagrams 等章节；逐段核对 `packages/mermaid/src/mermaid.ts` 中 `run` / `parse` / `render` / `initialize` 入口、`packages/mermaid/src/mermaidAPI.ts` 中 `renderDiagram` / `Diagram.fromText` / renderer / DOMPurify 路径；阅读官方 `mermaid-cli` README 的介绍、安装、调用与 Markdown 导出用法；对照官网 Getting Started、Usage、Flowchart、Sequence Diagram 的相关定义与示例。
- **通读：** 已完成
- **实现与安全边界：** 网页端由 `mermaid.run()` 查找未处理的 `.mermaid` 元素，调用 `mermaid.render()` 输出 SVG；`mermaidAPI` 调用解析与图形 renderer，常规安全级别对 SVG 做 DOMPurify 净化，sandbox 模式改在受限 iframe 内渲染。不应把用户自定义图文本默认视为可信 HTML。


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

## 代码运行入口核对（2026-11-12）

- **浏览器渲染：** `mermaid.initialize(config)` 配置后 `mermaid.run({querySelector: ".mermaid"})`；`mermaid.ts::runThrowsErrors` 查找未处理元素并调用 `render(id, txt, element)`，将返回 SVG 填入原 DOM。
- **公共 API：** `mermaid.ts::render` 将请求排入串行队列；调用 `mermaidAPI.render`。后者在 `mermaidAPI.ts` 处理配置、`Diagram.fromText` 解析类型、由对应 `diag.renderer.draw` 绘图，并序列化、处理输出安全级别。
- **安全边界：** 源码区分 `securityLevel === "sandbox"` 的 iframe 分支与默认 SVG sanitization 分支；`loose` 模式下安全属性有所不同，使用不可信输入应选择严格安全配置。
- **命令行导出：** 单独的 `@mermaid-js/mermaid-cli` 以 `mmdc -i input.mmd -o output.svg` 生成 SVG，也可输出 PNG/PDF、处理 Markdown 代码块。CLI 是独立仓库，不是主库 `mermaid.run` 自带的命令。
- **不同平台版本：** 文档站当前导航显示 12.x 版本；具体语法、兼容环境与安全配置须按部署版本查验，避免将官网新特性套用于旧版嵌入器。

### 官方源码链接

- [主库 README](https://github.com/mermaid-js/mermaid/blob/develop/README.md)
- [Web 入口：mermaid.ts](https://github.com/mermaid-js/mermaid/blob/develop/packages/mermaid/src/mermaid.ts)
- [底层渲染：mermaidAPI.ts](https://github.com/mermaid-js/mermaid/blob/develop/packages/mermaid/src/mermaidAPI.ts)
- [mermaid-cli README](https://github.com/mermaid-js/mermaid-cli)
