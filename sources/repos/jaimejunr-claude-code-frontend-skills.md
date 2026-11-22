# Claude Code Frontend Skills（JaimeJunr/claude-code-frontend-skills）

> 官方仓库资料归档；收录的是编排器和打包策略，不等于原始技能的独立改进证明。

- **标题：** Claude Code Frontend Skills — de-conflicted frontend marketplace
- **类型：** repo / Claude Code Marketplace + router + upstream 同步工具
- **链接：** https://github.com/JaimeJunr/claude-code-frontend-skills
- **入库日期：** 2026-11-22
- **核对范围：** 官方 README、`plugins/frontend-stack/skills/frontend-stack/SKILL.md`、`references/conflicts.md`、`stack.json` 及 `scripts/sync.py` 核心流程；未安装插件、未运行上游同步。
- **通读：** 已完成
- **代码：** https://github.com/JaimeJunr/claude-code-frontend-skills（公开；本仓 glue/脚本 MIT，各 vendor 插件保留各自许可）
- **wiki：** [Claude Code Frontend Skills](../../wiki/entities/claude-code-frontend-skills.md)

## 官方核心机制

- 一站式 marketplace 收录 Anthropic frontend-design、Impeccable、UI/UX Pro Max、Taste、Emil 的技能，另含 Hallmark 等可选项；`frontend-stack` 的 router 根据任务分配一个主导 Skill 和不超过两个辅助 Skill。
- `references/conflicts.md` 写明实际冲突裁决：高频产品交互按 Emil 短时/可中断动效，营销展示可由 Taste 主导；旧版风格 preset 只有显式指定才生效；`DESIGN.md` 归属和字体、配色冲突有确定优先级。
- `stack.json` 列出上游 repo/ref、复制路径、许可证和兼容性 patches；`scripts/sync.py` 按 ref fetch → 复制目录 → 打 patch → 写插件 manifest、`UPSTREAM.lock`、第三方 notice 与 README 技能表。
- 执行方式：`python3 scripts/sync.py --check` 是只校验工作树的入口；重新同步用 `python3 scripts/sync.py`，有外网与本地 git 依赖。
- README 提醒项目 `.claude/settings.json` 的 enabledPlugins **不代表已安装**；每台新机器仍需安装并开启新会话。周更 PR 受 GitHub token/CI 触发权限约束。

## 边界

整合方便但不自动保证零冲突；上游不断改变时需复查 patch 与版权清单。适合 Claude Code marketplace，不能直接推断同样的路由对其它 agent harness 生效。
