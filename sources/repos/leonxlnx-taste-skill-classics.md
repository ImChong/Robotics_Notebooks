# Taste Skill Classics（Leonxlnx/taste-skill-classics）

> 官方仓库资料归档；不是新版 Taste 的同一安装身份。

- **标题：** Taste Skill Classics — historical opt-in design presets
- **类型：** repo / Markdown Agent Skills 集合
- **链接：** https://github.com/Leonxlnx/taste-skill-classics
- **入库日期：** 2026-11-22
- **核对范围：** 官方完整 README、`skills/brutalist-skill/SKILL.md` 和 `skills/minimalist-skill/SKILL.md` 的风格、约束与调用方式；未实际渲染页面。
- **通读：** 已完成
- **代码：** https://github.com/Leonxlnx/taste-skill-classics（公开，MIT）
- **关系：** 2026 年 10 月从 [Taste Skill](../../wiki/entities/taste-skill.md) 分拆历史技能；新版主仓只保留 `design-taste-frontend`。
- **wiki：** [Taste Skill Classics](../../wiki/entities/taste-skill-classics.md)

## 原文要点

- 七项按名称显式启用：`high-end-visual-design`、`minimalist-ui`、`industrial-brutalist-ui`、`gpt-taste`、`stitch-design-taste`、`design-taste-frontend-v2-experimental`、`design-taste-frontend-v1`。
- `minimalist-ui` 约束克制单色、字体层次、平面 Bento 结构；`industrial-brutalist-ui` 定义 Swiss Industrial Print 与 Tactical Telemetry/CRT 两套视觉原型，并要求一次只选一个，禁止无意义的混搭。
- 历史 v1 / v2 experimental 用 1–10 的设计方差、动效强度与视觉密度旋钮；这些不代表当前主仓 v2.5 的默认流程。
- 示例安装：`npx skills add https://github.com/Leonxlnx/taste-skill-classics --skill "industrial-brutalist-ui"`；Claude Code / Codex 也有 plugin 安装方式。

## 局限

这里主要是 `SKILL.md` 文本设计规约，**无集中式可执行设计引擎**。应显式选风格且遵守目标品牌、可访问性和真实产品内容；视觉输出需要实际浏览器验收。
