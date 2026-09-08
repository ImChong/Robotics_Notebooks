# Diagram Design 项目页（cathrynlavery.github.io/diagram-design）

> 来源归档

- **标题：** Diagram Design — Editorial diagram gallery
- **类型：** site / project-page
- **URL：** <https://cathrynlavery.github.io/diagram-design/>
- **代码：** <https://github.com/cathrynlavery/diagram-design> — [`sources/repos/diagram-design.md`](../repos/diagram-design.md)
- **作者：** Cathryn Lavery
- **入库日期：** 2026-09-08
- **一句话说明：** 官方 GitHub Pages gallery：浏览 39 种 editorial 图表的 light / dark / full-editorial 变体，并链回 Claude Code / Codex / Pi 等安装说明。

## 开源核查（步骤 2.5，截至 2026-09-08）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **是** — 页内 CTA 与 README 均指向 `github.com/cathrynlavery/diagram-design` |
| 训练/推理入口 | **不适用**（文档/渲染 Skill，非 ML 框架） |
| 可运行实现 | **有** — 克隆后 `skills/diagram-design/assets/index.html` 本地可开；各 harness marketplace / `pi install` / symlink 安装 Skill |
| 数据 / 权重 | **无** |
| 综合判定 | **已开源**（MIT） |

## 页面要点（2026-09-08 README / homepage）

- Hero：39 editorial diagram types；自包含 HTML + SVG；**No shadows. No Mermaid slop.**
- Gallery：按类型切换 **minimal light / minimal dark / full-editorial**；与仓内 `assets/index.html` 同构。
- 安装：Claude Code plugin marketplace、Codex plugin、Pi `install`、Factory Droid、Kiro import URL、OpenCode symlink 等（详见 README Install 节）。
- 子资源：`docs/cookbook.md`（可编辑安装、品牌 onboarding、import/export 配方）、`docs/screenshots/` 示例图。

## 关联资料

- 仓库归档：[`sources/repos/diagram-design.md`](../repos/diagram-design.md)
- Wiki：[`wiki/entities/diagram-design.md`](../../wiki/entities/diagram-design.md)
