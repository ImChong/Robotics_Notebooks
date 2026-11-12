## [2026-11-12] ingest | sources/repos/nutlope-hallmark.md — 收录 Hallmark 前端设计 Agent Skill 与源码检查链

- 意图：将用户提供的 Nutlope/hallmark 原始仓库转换为可交叉引用的工具实体知识页。
- 核对：通读官方 README 与 SKILL.md，核查主题选择和 sloplint 的源码入口；官网确认安装命令 `npx skills add nutlope/hallmark -g`。
- 开源结论：MIT；技能文档、50 个主题目录和 Node.js 检查脚本公开；浏览器渲染检查另需 Chrome 与 puppeteer-core。
- 写回：新增 [Hallmark](../wiki/entities/hallmark.md)，关联 [MengTo/Skills](../wiki/entities/mengto-skills.md)。
- 验证边界：已核对源码与官方文档，未实际安装执行，也未运行完整知识库 CI。
