---
type: entity
project_id: answer-me-with-html
project: https://answer-me-with-html.com/
code: https://github.com/QingYunA/answer-me-with-html
tags:
  - llm-agents
  - coding-agents
  - skills
  - html
  - visual-explanation
status: complete
updated: 2026-10-17
related:
  - ./agent-skills-addyosmani.md
  - ./vercel-agent-browser-skill.md
  - ../concepts/agentic-coding-software-fundamentals.md
  - ../references/llm-wiki-karpathy.md
sources:
  - ../../sources/repos/qingyuna-answer-me-with-html.md
  - ../../sources/sites/answer-me-with-html.md
summary: "Answer with HTML 是面向编码代理的 Agent Skill：模型写简短 Markdown 草稿，随包 CLI 负责排版和图表渲染，生成便于阅读、可离线打开的单页 HTML。"
---

# Answer with HTML（一页能读的答案）

**Answer with HTML** 是一个面向编码代理的开源 Skill：模型用 Markdown 写答案内容，随包 CLI 自动生成包含布局、样式和图表的单页 HTML。

## 一句话定义

它把“写什么”交给模型，把 HTML、CSS、SVG 图表排版交给本地 CLI，以较短草稿生成适合阅读和离线分享的解释页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HTML | HyperText Markup Language | CLI 生成的单页文档格式 |
| CLI | Command-Line Interface | 接收 Markdown 草稿并渲染页面的命令行程序 |
| SVG | Scalable Vector Graphics | 页面图表与流程示意使用的矢量图形格式 |
| STE100 | Simplified Technical English ASD-STE100 | 项目提供的简明写作检查规则 |

## 为什么重要

复杂技术答复常需要流程图、结构图、对比表和多个段落。直接让模型手写 HTML 会让大量输出 token 花在标签、CSS 和 SVG 坐标上。这个 Skill 将内容和呈现拆开：模型写短 Markdown，CLI 负责可重复的页面布局。

对机器人知识整理而言，它适合把架构解释、算法比较、实验流程或排查结论做成易读页面；它本身不提供机器人专用数据、控制算法或仿真能力。

## 核心原理

| 阶段 | 处理内容 |
|------|----------|
| 1. 判断形式 | Agent 按 Skill 指引判断答案是否适合页面呈现；简单的一句话问题可直接回答 |
| 2. 写内容草稿 | Agent 以用户语言写 Markdown；标题与列表构成页面分区 |
| 3. 本地渲染 | 随包 CLI 自动处理 HTML/CSS/SVG 布局、主题与图表 |
| 4. 交付页面 | 输出单个 HTML 文件，可离线查看；页面可保留 Markdown 源文以便继续编辑 |

README 还描述了解释视频模式。具体图表组件、主题、修补现有页面和配置项以仓库的 Reference 为准。

## 工程实践

| 项目 | 说明 |
|------|------|
| 安装 | `npx skills add QingYunA/answer-me-with-html`；需要 Node.js 20+ |
| 直接渲染 | `node skills/answer-me-with-html/scripts/am.mjs render draft.md` |
| 输出目录 | 默认保存到 `~/.answer-me-with-html/pages/`；可用 `AM_HOME` 更改 |
| 可复核的性能说法 | 官方 README 报告：9 个手写页面平均 4,893 个输出 token，对比 Skill 草稿平均 612 个；小规模测试的中位耗时是 31 秒与 12 秒。应视为项目方报告值，不能直接外推到不同模型、问题或环境 |

建议先对照 README 的示例，检查生成页是否保留必要的技术限定、来源、单位和不确定性，再分享 HTML 文件。视觉排版不能替代技术事实核查。

## 局限与风险

- 项目是通用 Agent 答案渲染工具，不是机器人领域的知识库、仿真器或策略评测套件。
- 简短草稿会压缩内容；若模型遗漏假设、数据出处或失败条件，漂亮的页面也会放大误解。重要结论需要核对草稿与来源。
- Token 和耗时数据来自 README 中的项目方基准测试，不是独立评测；运行结果受模型、任务、环境和输出复杂度影响。
- 官网页面正文在本次 ingest 中未能读取；可用实现、安装和许可证信息已通过官方 GitHub 仓库核实。

## 关联页面

- [Agent Skills（Addy Osmani）](./agent-skills-addyosmani.md) — 将软件工程流程打包成编码代理技能
- [agent-browser（Vercel Labs Skill）](./vercel-agent-browser-skill.md) — 让代理通过 CLI 操作浏览器
- [Agentic Coding 时代的软件工程基础](../concepts/agentic-coding-software-fundamentals.md) — 人与编码代理的协作边界
- [Karpathy LLM Wiki](../references/llm-wiki-karpathy.md) — 把检索与对话结论沉淀为可复用知识

## 参考来源

- [项目仓库归档（本站）](../../sources/repos/qingyuna-answer-me-with-html.md)
- [官网核查记录（本站）](../../sources/sites/answer-me-with-html.md)
- [官方 GitHub 仓库](https://github.com/QingYunA/answer-me-with-html)
- [官方使用参考](https://github.com/QingYunA/answer-me-with-html/blob/main/docs/reference.md)

## 推荐继续阅读

- [官方 README](https://github.com/QingYunA/answer-me-with-html#readme) — 安装、工作流、基准方法与示例
- [官方中文 README](https://github.com/QingYunA/answer-me-with-html/blob/main/README.zh-CN.md)
