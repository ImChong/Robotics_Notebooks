# Agent Skills（addyosmani/agent-skills）官方文档通读

> 来源归档 · 原文通读 · 对应原有 wiki 实体

- **标题：** Agent Skills — Production-grade engineering skills for AI coding agents
- **类型：** repo / 官方技术文档
- **作者：** Addy Osmani 及仓库贡献者
- **原文：** <https://github.com/addyosmani/agent-skills/blob/main/README.md>
- **辅助一手文档：** <https://github.com/addyosmani/agent-skills/blob/main/docs/getting-started.md>
- **核对日期：** 2026-11-18
- **核对范围：** 通读官方 README 的 Commands、Quick Start、Adoption、25 Skills、Agent Personas、Reference Checklists、How Skills Work、Project Structure、Comparison、Contributing 等章节，并对照 docs/getting-started.md 的完整安装与技能加载规则；未逐一测试所有宿主环境，也未执行技能本身。
- **通读：** 已完成
- **源码/开放状态：** GitHub 公开，MIT；面向代理的 Markdown 技能与命令包装，并非机器人控制器或独立模型权重。
- **映射：** [Agent Skills wiki 实体](../../wiki/entities/agent-skills-addyosmani.md)

## 从原文抽取的关键事实

1. **生命周期命令：** `/spec`、`/plan`、`/build`、`/test`、`/constraints`、`/review`、`/webperf`、`/code-simplify`、`/ship`；一条可选 `/build auto` 在批准计划后逐任务实施，不能被解释为跳过验证。
2. **技能组织：** 25 个 skills 包含 `using-agent-skills` 元技能；`agents/` 含 code reviewer、test engineer、security auditor、web performance auditor 四个审查 persona；`references/` 为共用质量检查表。
3. **标准路径：** 需求规范 → 原子任务规划 → 增量实现与 TDD → 测试/调试 → Review/安全/性能 → 上线检查；`SKILL.md` 明确触发条件、步骤、验证与常见借口。
4. **安装边界：** `npx skills add addyosmani/agent-skills` 可安装；但单独安装某个 skill 时，仓库级 `references/` 不随之拷贝，需手动补齐或整仓集成（README 链接 issue #361）。
5. **运行方式：** `SKILL.md` 是加载到 agent 上下文的工作流，不是受编译器强制执行的机器人安全证明；目标仓库仍必须运行自己的 lint、测试与 CI。

## 归纳与局限

- 该包解决的是软件交付约束覆盖面；与轻量可组合的 [mattpocock/skills](mattpocock-skills.md) 和跨运行时的 [pstack-claude](michael-denyer-pstack-claude.md) 的关注重点不同。
- README 中跨 Agent 环境的兼容性描述是作者说明，本文未做实测；具体安装应按相应 `docs/*-setup.md` 执行。
