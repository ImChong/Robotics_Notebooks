# Answer me with HTML（QingYunA/answer-me-with-html）

> 来源归档（ingest）

- **标题：** Answer me with HTML
- **类型：** GitHub 仓库 / Agent Skill
- **作者：** QingYunA
- **仓库：** <https://github.com/QingYunA/answer-me-with-html>
- **官网：** <https://answer-me-with-html.com/>
- **分发：** `npx skills add QingYunA/answer-me-with-html`
- **运行要求：** Node.js 20+
- **源码状态：** 已开源；仓库包含 Skill、CLI 源码与随包分发的 CLI
- **许可证：** MIT
- **入库日期：** 2026-10-17
- **一句话说明：** 面向编码代理的 Skill：模型撰写简短 Markdown 答案草稿，随 Skill 提供的 CLI 自动排版并渲染成可离线查看的单页 HTML，支持图表、流程与解释视频模式。
- **为什么值得保留：** 展示了将内容生成与 HTML/SVG/CSS 排版拆开的 Agent 工作流，可供复杂技术解释、对比和流程说明使用；它是知识表达工具，不是机器人算法或仿真框架。
- **沉淀到 wiki：** 是 → [Answer with HTML](../../wiki/entities/answer-me-with-html.md)

## README 要点（归纳）

- **工作流：** Agent 先写扩展 Markdown 草稿，再调用随包 CLI 渲染；CLI 负责页面布局、主题、样式与图表，输出为单个 HTML 文件。
- **交互：** 渲染页包含主题切换、复制/收集 Markdown 等控件；页面保留生成它的 Markdown 源文。
- **图表与视频：** 文档展示了流程图、状态图、时间线、比较表等组件，也提供解释视频模式。
- **本地运行：** Skill 内含 `scripts/am.mjs`；无需单独 `npm install`，需要 Node.js 20+。可直接运行 `node skills/answer-me-with-html/scripts/am.mjs render <draft.md>`。
- **安装入口：** README 推荐 `npx skills add QingYunA/answer-me-with-html`；同时提供 Claude Code 插件及其他 Agent 的手动安装说明。
- **性能数据（作者自测）：** README 报告 9 个手工 HTML 样例平均 4,893 个输出 token，使用 Skill 后 Markdown 草稿平均 612 token（约 8 倍差异）；同一小型测试报告中位耗时 31 秒对 12 秒（约 2.6 倍）。这些是项目方在其指定模型与样例上的测量，不代表独立复现或所有问题上的结果。

## 关联资料

- [官网核查](../sites/answer-me-with-html.md)
- [知识页](../../wiki/entities/answer-me-with-html.md)
- [官方 README](https://github.com/QingYunA/answer-me-with-html#readme)
- [CLI 使用参考](https://github.com/QingYunA/answer-me-with-html/blob/main/docs/reference.md)
