# UI/UX Pro Max（nextlevelbuilder/ui-ux-pro-max-skill）

> 官方仓库资料归档；以下为技术摘要，不转载上游全文。

- **标题：** UI/UX Pro Max — Design Intelligence
- **类型：** repo / Agent Skill + Python 搜索脚本
- **链接：** https://github.com/nextlevelbuilder/ui-ux-pro-max-skill
- **入库日期：** 2026-11-22
- **核对范围：** 官方 README、`.claude/skills/ui-ux-pro-max/SKILL.md`、`src/ui-ux-pro-max/scripts/search.py` 的命令行参数、检索与设计系统入口；核对其安装与输出约定，未在目标应用执行 UI 检查。
- **通读：** 已完成
- **代码：** https://github.com/nextlevelbuilder/ui-ux-pro-max-skill（公开，MIT）
- **项目定位：** 从本地可搜索样式、产品配色、排版、UX、图表和技术栈规则中检索适用指南，输出设计系统与 Token，而非调用专用设计生成模型。
- **wiki：** [UI/UX Pro Max](../../wiki/entities/ui-ux-pro-max-skill.md)

## 核心机制与证据

- 上游 `SKILL.md` 自述数据：80 个可检索风格（其中 51 active）、194 类产品配色/推理资料、74 种字体组合、141 条 UX 指南、105 个图标、17 个 GSAP preset、25 种图表类型及 22 种技术栈；均是**上游版本快照宣称**，非独立复测。
- 检索契约：新页面/整站使用 `--design-system`；单项问题使用 `--domain`；实现特定框架使用 `--stack`。搜索无匹配要收紧查询再试，不得编造命中。
- `src/ui-ux-pro-max/scripts/search.py` 导入 `core.search` / `search_stack` 和 `design_system.generate_design_system`，支持 `--tokens dtcg|css`、`--persist`、`--preview`；输出支持设计 Token，可持久化到 `design-system/<project>/MASTER.md`。
- 安装：Claude Code 可通过 marketplace；CLI 示例 `npx ui-ux-pro-max-cli init --ai claude`。具体目录路径取决于安装方式；不能假定应用已加载技能。

## 边界

设计知识库返回的是推荐，不等于 WCAG 或品牌验收；对比度、键盘焦点、长文案、移动端触控仍需跑真实页面、截图与设备检查。仓库虽公开，但依赖环境（Python 3）和输入上下文。
