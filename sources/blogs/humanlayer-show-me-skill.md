# HumanLayer：Show Me Skill — Make it visual

> 来源归档

- **标题：** Show Me Skill: Make it visual
- **类型：** blog
- **作者：** HumanLayer
- **链接：** https://www.humanlayer.com/blog/show-me-skill
- **发布日期：** 2026-08-12（页面标注）
- **对应仓库：** [humanlayer/skills](https://github.com/humanlayer/skills)，当前核查 HEAD [653b641](https://github.com/humanlayer/skills/commit/653b6411c1f70c275a18e37673b042ff99f67ceb)
- **协议：** 仓库 MIT；博客文章为说明材料，遵循网站自身条款
- **一句话说明：** 介绍 `show-me` Agent Skill：当文字解释不够直观时，要求 agent 选合适的轻量图示或 HTML 讲解成品，并避免过度叙述。
- **为什么值得保留：** 将「解释当前话题」转成面向读者的可视化交付，覆盖伪代码、调用栈、组件结构、浅文件树、Mermaid、HTML mockup/交互说明，能提升架构讨论与代码讲解的可读性。

## 文章要点

- 可视形式应由问题决定：执行顺序适合调用栈/序列图，层级适合树状图，布局适合 HTML mockup；目标是降低理解成本，而不是为了有图而有图。
- 适用例子包括代码路径、组件树、数据流、仓库浅层结构，以及将概念做成可运行/可交互的单页解释器。
- 这是给 agent 的工作指南，不是绘图引擎；HTML artifact 的准确性、可访问性及是否符合仓库技术栈仍由 agent/使用者验证。

## 映射到 Wiki

- [HumanLayer Skills](../../wiki/entities/humanlayer-skills.md) — 将该技能与同仓库其他 Skills 一起理解；不另建重复实体。
- [上游 `show-me` Skill](https://github.com/humanlayer/skills/tree/main/plugins/show-me/skills/show-me) — 实际安装与调用内容以当前技能文件为准。
