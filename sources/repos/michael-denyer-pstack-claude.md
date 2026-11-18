# pstack for Claude Code and other harnesses（michael-denyer/pstack-claude）

> 官方仓库来源归档 · 原文核对

- **标题：** pstack — 多 Agent 环境中的严谨工程工作流移植
- **类型：** repo
- **作者/维护者：** Michael Denyer（移植版）；原始 pstack 作者 Lauren Tan
- **仓库：** <https://github.com/michael-denyer/pstack-claude>
- **原始 pstack：** <https://github.com/cursor/plugins/tree/main/pstack>
- **官方文档：** <https://github.com/michael-denyer/pstack-claude/blob/main/docs/reference.md>
- **插件说明：** <https://github.com/michael-denyer/pstack-claude/blob/main/plugins/pstack/README.md>
- **许可：** 移植与附加内容 MIT；移植所含上游作品另见 NOTICE 与原许可
- **核对日期：** 2026-11-18
- **核对范围：** 通读 README、plugins/pstack/README.md，核对 docs/reference.md 的命令列表、宿主适配与配置章节、plugins/pstack/skills/poteto-mode/SKILL.md 的主工作流、plugins/pstack/skills/statechart/SKILL.md，以及实际源码 plugins/pstack/skills/statechart/scripts/statechart.ts 与 worker.ts；未逐条审阅全部下属原则技能，未在本地运行 TLA+/插件集成测试。
- **通读：** 已完成
- **开放状态：** 已开源，公开 Markdown skills、plugin hooks、Pi 扩展与本地脚本；无独立模型权重。
- **wiki 映射：** [pstack-claude 独立详情节点](../../wiki/entities/michael-denyer-pstack-claude.md)

## 核心内容（依据官方文档）

1. **身份：** 本仓库是 Lauren Tan 原版 pstack 的跨宿主移植，不是重新发布一个机器人训练框架；将 Cursor 的原语映射到 Claude Code、Codex、Pi、Copilot、Muse、OpenCode 等宿主。
2. **统一入口：** `poteto-mode` 根据需求调用专门 playbook：调查 `how`/`why`、方案 `architect`、并行候选 `arena`/`swarm`、对抗性评审 `interrogate`、去赘述 `unslop`、验证与 PR 维护等。
3. **运行时组件：** `plugins/pstack/skills/` 存储 Markdown 技能和 principle-*；`plugins/pstack/hooks/` 提供 SessionStart 等适配；`plugins/pstack/pi/index.ts` 为 Pi 提供子代理/提问/唤醒能力。跨宿主工具名需要映射，技能不能自动获得原宿主缺失的权限。
4. **可执行实例：** `statechart` skill 接收一个 TLA+ 规格和 .cfg，可传 view 模块；`statechart.ts` 校验参数、建立隔离 staging 目录并 spawn `worker.ts`；worker 调用 `run.ts` 内的 TLC 检查及 `pipeline.ts` 编译，写入 report.md / chart.mmd 等文件；主进程验证完成后将暂存结果原子移动到输出目录。
5. **安全/数据：** README 声明没有 pstack 自有服务或遥测，但代理读取的对话、上下文会传给其模型服务商；脚本在本地运行，PR 相关工具使用本机 GitHub CLI 凭据。决策 trail 默认不提交，共享前应去敏。
6. **版本边界：** 宿主兼容性和命令数量会变化，以 README、docs/reference.md 与实际目录为准；不可将“支持插件”解读成全部平台每项功能均实测可用。

## 对照阅读

- [Agent Skills（Addy Osmani）](addyosmani-agent-skills.md)：全 SDLC 的独立 skills 与质量门禁。
- [Skills For Real Engineers（Matt Pocock）](mattpocock-skills.md)：轻量可组合工程习惯与术语文档。
- [Lauren Tan 演讲笔记](../videos/lauren-tan-all-code-is-slop-2026.md)：原始 pstack 的状态迁移检查理念。
