---
type: entity
project_id: drawnix
project: https://drawnix.com/
code: https://github.com/plait-board/drawnix
tags: [visualization, diagramming, whiteboard, mind-map, flowchart, mermaid, open-source]
status: complete
updated: 2026-10-09
related:
  - ./mermaid-js.md
  - ./gitdiagram.md
  - ./archify.md
sources:
  - ../../sources/repos/drawnix.md
  - ../../sources/sites/drawnix-com.md
summary: "Drawnix 是基于 Plait 的开源白板应用，支持思维导图、流程图、自由绘图、Mermaid / Markdown 转画布、浏览器保存与图像 / JSON 导出。"
---

# Drawnix（开源一体化白板）

**Drawnix** 是一个基于 Plait 插件框架的开源白板应用：用户可在同一无限画布上制作思维导图、流程图和自由绘图，并把 Mermaid / Markdown 内容转换为可编辑的画布元素。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SaaS | Software as a Service | 仓库将 Drawnix 描述为可在线使用的白板产品 |
| JSON | JavaScript Object Notation | .drawnix 文件用于保存或导出结构化画板内容 |
| PNG | Portable Network Graphics | README 列出的画布图像导出格式 |
| UI | User Interface | React 画板界面承载工具栏、编辑交互和插件能力 |
| MIT | Massachusetts Institute of Technology License | 仓库根目录采用的宽松开源许可证 |

## 为什么值得关注

机器人研究与工程会反复处理控制流程、系统架构、实验计划和动作层级。Drawnix 提供一个把结构图与自由批注放在同一画布上的编辑器；对机器人学习者，它适合把 Mermaid 草图继续编辑成图形、把 Markdown 大纲变成思维导图，或在方案评审中快速组合流程与图片。

它的价值在于**将结构化文本和视觉编辑接起来**：Mermaid / Markdown 是快速生成起点，白板是人工整理和补充空间。它不是机器人仿真、控制或知识图谱系统。

## 核心能力与数据流

| 能力 | 输入 | 输出 / 结果 | 依据 |
|------|------|-------------|------|
| 思维导图、流程图、自由绘图 | 用户操作 | 无限画布中的图形元素 | README 特性列表 |
| Mermaid 转白板元素 | Mermaid 定义文本 | 可插入画布的 Plait 元素 | Mermaid 转换组件与 README |
| Markdown 转思维导图 | Markdown 文本 | 画布中的层级节点 | README 与对应转换依赖 |
| 自动保存 | 画板内容与偏好 | 浏览器 IndexedDB，无法使用时回退 LocalStorage | apps/web/src/app/app.tsx |
| 导出 | 当前画板 | PNG / JSON（英文 README 还列 JPG） | 中英文 README |

官方 README 说明应用以 Plait 为底层绘图框架，并采用插件机制组织功能。仓库包清单还包含 @plait-board/mermaid-to-drawnix、@plait-board/markdown-to-drawnix、Plait 核心绘图包、Slate 富文本组件与 React 视图层。

~~~mermaid
flowchart LR
  AUTHOR["研究者输入 Mermaid / Markdown"] --> CONVERT["对应转换插件解析"]
  CONVERT --> ELEMENTS["生成 Plait 画布元素"]
  ELEMENTS --> CANVAS["React 无限画布编辑"]
  CANVAS --> STORE["浏览器本地持久化"]
  CANVAS --> EXPORT["PNG / JPG / .drawnix JSON"]
~~~

## Mermaid 到画布的运行时序

源码中的 Mermaid 对话框延迟加载转换模块，随着输入变化调用 parseMermaidToDrawnix，生成元素后由用户插入画布；画板变化回调将内容写入 localForage 配置的浏览器存储。这个序列依据 mermaid-to-drawnix.tsx 与 apps/web/src/app/app.tsx 描述。

~~~mermaid
sequenceDiagram
    autonumber
    actor Author as 研究者
    participant Web as Drawnix Web App
    participant Parser as Mermaid 转换插件
    participant Board as Plait 画板
    participant Store as IndexedDB / LocalStorage
    participant Export as 导出功能

    Author->>Web: 打开 Mermaid 转换对话框
    Web->>Parser: 延迟加载转换模块
    Author->>Web: 输入 Mermaid 流程图定义
    Web->>Parser: parseMermaidToDrawnix(definition)
    Parser-->>Web: 返回画布元素
    Web->>Board: 用户确认后插入元素
    Board-->>Web: onChange 返回画板内容
    Web->>Store: 保存画板与工具状态
    Author->>Export: 选择导出格式
    Export-->>Author: PNG / JPG / .drawnix JSON
~~~

### 本地开发入口

在仓库根目录执行：

~~~bash
npm install
npm run start
~~~

根目录 package 脚本将 start 映射为 nx serve web --host=0.0.0.0。Docker 用户可参考官方 README 的 pubuzhixing/drawnix:latest 镜像入口；镜像标签可变，部署前应核对所用版本。

## 开源状态与局限

- **已开源，MIT。** 根目录 LICENSE 给出 MIT 文本，仓库同时提供可运行的 Web 应用源码。许可证适用于代码再利用，不代表托管服务本身具有相同服务条款。
- **本地保存边界清楚。** 当前 Web App 将画板内容、工具状态和偏好写入 IndexedDB / LocalStorage；不要把浏览器自动保存理解成账号云同步或异地备份。重要画板应主动导出并保管。
- **转换不等于无损往返。** Mermaid / Markdown 输入会转换成白板元素。复杂语法、布局细节、不同 Mermaid 图型的支持程度需在目标版本中验证；编辑后的画板是否能还原为原始文本也不能默认保证。
- **通用白板，不是机器人专用软件。** 它不能代替代码评审、仿真器、CAD 或正式实验记录系统；机器人项目中的流程图应继续与代码、参数和测试证据关联。
- **在线服务状态需单独核验。** 仓库列出 drawnix.com 作为在线应用入口；本次检索没有取得网页正文，因此不据此推断云同步、协作或服务可用性。

## 使用建议

1. 用 Mermaid 起草系统的数据流或策略运行流程；若需进一步排版，可转换进画布，再手动调整布局、颜色与注释。
2. 把 Markdown 学习大纲导入为思维导图，按控制、学习、部署等主题分组；保留原 Markdown 作为版本可追踪的源文件。
3. 对机器人方案评审，将关键节点连回 GitHub、论文、仿真配置和测试记录；白板图只承载概览，不作为唯一事实来源。
4. 跨设备迁移前导出 .drawnix / JSON 或图像，先确认导入与恢复流程。

## 关联页面

- [Mermaid.js](./mermaid-js.md) — 可版本管理的文本式图表语言与渲染库；Drawnix 提供转成可编辑画布元素的路径
- [GitDiagram](./gitdiagram.md) — 根据仓库结构证据生成交互式代码架构图，适合浏览源码关系
- [Archify](./archify.md) — 将结构化 JSON 图描述编译成可校验的架构、工作流和时序图

## 参考来源

- [Drawnix 仓库来源档案](../../sources/repos/drawnix.md)
- [Drawnix 官方在线应用入口档案](../../sources/sites/drawnix-com.md)
- [GitHub README](https://github.com/plait-board/drawnix/blob/develop/README.md)
- [Web App 保存实现](https://github.com/plait-board/drawnix/blob/develop/apps/web/src/app/app.tsx)
- [Mermaid 转换组件](https://github.com/plait-board/drawnix/blob/develop/packages/drawnix/src/components/ttd-dialog/mermaid-to-drawnix.tsx)
- [MIT License](https://github.com/plait-board/drawnix/blob/develop/LICENSE)

## 推荐继续阅读

- [官方在线应用](https://drawnix.com/) — 试用入口；当前服务与账号功能以实际页面为准
- [Plait 绘图框架](https://github.com/plait-board/plait) — Drawnix 使用的底层绘图与插件框架
- [Mermaid 转换包](https://github.com/plait-board/mermaid-to-drawnix) — Mermaid 定义到 Drawnix 画布元素的转换器
- [Markdown 转思维导图包](https://github.com/plait-board/markdown-to-drawnix) — Markdown 文本转换入口
