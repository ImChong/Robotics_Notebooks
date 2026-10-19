# book-to-skill（virgiliojr94/book-to-skill）

> 来源归档

- **标题：** book-to-skill
- **类型：** repo
- **作者：** Virgilio Jr.
- **代码：** <https://github.com/virgiliojr94/book-to-skill>
- **默认分支：** master
- **入库日期：** 2026-10-19
- **一句话说明：** 将技术书籍、文档目录或研究资料集合转换为符合开放 Agent Skills 标准的技能包，生成核心指引、章节文件、术语表、模式集与速查表，供代理按需读取。
- **为什么值得保留：** 可把机器人控制、强化学习、仿真等领域的技术书籍和内部文档编译成可复用的代理工作上下文；章节按需加载，减少每次从头翻阅整份材料的成本。它与本库的 LLM Wiki 互补：前者面向特定资料生成可安装技能，后者由人持续策展跨来源、可交叉引用的知识页。README 所称 24×–51× token 节省是上游基准结果，不代表任意书籍或问题都能达到同等比例。
- **源码状态：** GitHub 仓库公开可读；README 徽章标注 MIT License。
- **沉淀到 wiki：** 是 → [book-to-skill 实体页](../../wiki/entities/book-to-skill.md)

## README 要点（归纳）

- **输入：** 技术书籍、单个文件、目录、glob 或多份资料；README 列出 PDF、EPUB、DOCX、Markdown、HTML、RTF 与 MOBI 等格式。
- **生成物：** `SKILL.md`（核心框架与章节索引）、按章拆分的内容文件、`glossary.md`、`patterns.md` 与 `cheatsheet.md`。
- **使用方式：** 默认生成跨代理用户级 skill，兼容支持开放 Agent Skills 标准的工具；也可选择项目本地或特定代理目录。
- **主要取舍：** 按需加载可减少上下文重复，但生成内容仍需回到原书或原始文档核验；README 的性能比例来自其选定书籍与测量方法，不能视作普适保证。

## 对 wiki 的映射

- 主实体：[book-to-skill](../../wiki/entities/book-to-skill.md)
- 对照阅读：[LLM Wiki（Karpathy 模式）](../../wiki/references/llm-wiki-karpathy.md) — 持久、跨来源的人工策展知识层。
