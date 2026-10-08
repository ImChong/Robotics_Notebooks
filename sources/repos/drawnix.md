# Drawnix（plait-board/drawnix）

> 来源归档（ingest）

- **类型：** repo / whiteboard / diagramming / visualization
- **主仓库：** <https://github.com/plait-board/drawnix>
- **开发分支：** develop（归档核查时的 GitHub 默认分支）
- **官方在线应用：** <https://drawnix.com/>
- **许可证：** MIT（根目录 LICENSE，版权标注 Drawnix 2024）
- **入库日期：** 2026-10-09
- **一句话说明：** Drawnix 是基于 Plait 的开源白板应用，把思维导图、流程图、自由绘图和图片放在同一无限画布，并支持 Mermaid / Markdown 转画布、浏览器保存和图像 / JSON 导出。

## 一手资料

| 来源 | 用途 |
|------|------|
| [中文 README](https://github.com/plait-board/drawnix/blob/develop/README.md) | 产品定位、特性、Plait 插件架构、开发和 Docker 启动方式 |
| [English README](https://github.com/plait-board/drawnix/blob/develop/README_en.md) | 特性与模块结构补充；声明支持 PNG、JPG 和 .drawnix JSON 导出 |
| [根目录 LICENSE](https://github.com/plait-board/drawnix/blob/develop/LICENSE) | MIT 许可文本 |
| [package.json](https://github.com/plait-board/drawnix/blob/develop/package.json) | Nx 工作区、脚本和主要依赖 |
| [Web App app.tsx](https://github.com/plait-board/drawnix/blob/develop/apps/web/src/app/app.tsx) | IndexedDB / LocalStorage 持久化、画板载入和变更保存 |
| [Mermaid 转换组件](https://github.com/plait-board/drawnix/blob/develop/packages/drawnix/src/components/ttd-dialog/mermaid-to-drawnix.tsx) | Mermaid 文本解析为元素并插入当前画板的调用路径 |

## 仓库要点

- README 所列能力包括思维导图、流程图、画笔、图片、无限画布、主题与移动端适配、撤销 / 重做、浏览器自动保存，以及 Mermaid 语法转流程图和 Markdown 转思维导图。
- 中文 README 当前列出 PNG 与 .drawnix JSON 导出；英文 README 还列出 JPG。导出格式以实际版本界面和 README 为准。
- 产品以 Plait 绘图框架为核心，采用可扩展插件组织方式。README 的仓库结构列出 apps/web、packages/drawnix、packages/react-board 与 packages/react-text。
- Web 应用的 app.tsx 使用 localForage，并优先配置 IndexedDB、回退 LocalStorage；画板内容、工具状态与偏好保存在浏览器端。
- Mermaid 转换对话框动态加载 @plait-board/mermaid-to-drawnix，解析定义并将生成的元素插入当前画布。它是可编辑画布转换路径，不应假设支持所有 Mermaid 图型或能无损往返。
- 根目录 package.json 的开发脚本为 npm install 后运行 npm run start（底层为 nx serve web --host=0.0.0.0）；仓库也提供 Docker 镜像入口 pubuzhixing/drawnix:latest。
- 顶层 package 标记 private: true，不应把仓库误解为发布在 npm 的单一库包。

## 开源状态与边界

| 项目 | 核查结果 |
|------|----------|
| 应用源码 | **已开源** — 公开仓库，MIT |
| 数据 / 权重 | **不适用** — 这是白板应用，不是模型或数据集 |
| 在线应用 | README 链接至 <https://drawnix.com/> 并称其为最小化应用；入库时浏览器搜索未取得可读取的站点正文，功能描述以仓库 README 和源码为依据 |
| 保存与同步 | 源码确认自动保存位于浏览器 IndexedDB / LocalStorage；不据此推断有账号云同步或多人实时协作 |

## Wiki 映射

- [Drawnix 项目页](../../wiki/entities/drawnix.md)
- [Mermaid.js](../../wiki/entities/mermaid-js.md) — Mermaid 文本在 Drawnix 中可转换为可编辑的白板元素
