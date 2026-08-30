# Archify 项目页（tt-a1i.github.io/archify）

> 来源归档

- **标题：** Archify — Technical Diagrams from Plain English
- **类型：** site / project-page
- **URL：** <https://tt-a1i.github.io/archify/>
- **代码：** <https://github.com/tt-a1i/archify> — [`sources/repos/archify.md`](../repos/archify.md)
- **作者：** tt-a1i
- **入库日期：** 2026-08-30
- **一句话说明：** 官方静态站：安装命令、五类图、四套视觉预设、Proof Lab 已校验工件，以及 Cursor / Codex / Claude Code / OpenCode 的 agent-aware 快速开始。

## 开源核查（步骤 2.5，截至 2026-08-30）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **是** — 页脚与 CTA 指向 `github.com/tt-a1i/archify` |
| 训练/推理入口 | **不适用**（文档/渲染工具，非学习框架） |
| 可运行实现 | **有** — `npx skills add tt-a1i/archify`；仓内 `node archify/bin/archify.mjs doctor\|validate\|deliver`；Proof Lab 11 个已入库场景 |
| 数据 / 权重 | **无**（无模型权重；图源是 typed JSON IR） |
| 综合判定 | **已开源**（MIT） |

## 页面要点（2026-08-30 抓取）

- Hero：从自然语言（或仓库描述）生成可探索 HTML 图；开发号 **v2.16.0-dev.0**。
- 五类图：Architecture / Workflow / Sequence / Data Flow / Lifecycle。
- 四套视觉身份：Classic、Signal Flow、Blueprint、Editorial；深浅主题成对切换。
- 导出：4× 原生栅格（PNG / JPEG / WebP）、双主题 SVG、WebM、剪贴板 PNG。
- 安装：`npx skills add tt-a1i/archify -g`；Cursor 非交互示例：`npx -y skills add tt-a1i/archify --skill archify --agent cursor --global --copy --yes`。
- 子页：[`guide.html`](https://tt-a1i.github.io/archify/guide.html) 场景向导；[`gallery.html`](https://tt-a1i.github.io/archify/gallery.html) Proof Lab；[`start.html`](https://tt-a1i.github.io/archify/start.html?agent=cursor&type=architecture) agent 切换器。
- 赞助商区（APINEBULA / EverMind Raven）与产品能力无关，归档不展开。

## 关联资料

- 仓库归档：[`sources/repos/archify.md`](../repos/archify.md)
- Wiki：[`wiki/entities/archify.md`](../../wiki/entities/archify.md)
