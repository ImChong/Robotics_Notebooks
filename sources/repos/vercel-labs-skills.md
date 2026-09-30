# skills CLI（vercel-labs/skills）

> 来源归档

- **标题：** skills — The CLI for the open agent skills ecosystem
- **类型：** repo
- **链接：** https://github.com/vercel-labs/skills
- **分发：** https://skills.sh/vercel-labs/skills（npm 包 `skills`）
- **入库日期：** 2026-09-30
- **代码：** **已开源**（MIT；CLI 源码、`skills/find-skills` 元技能与 agent 适配逻辑均在 GitHub）
- **一句话说明：** 开放 Agent Skills 生态的包管理器：`npx skills add/find/update/use`，把 GitHub/GitLab/本地路径上的 `SKILL.md` 安装到 Claude Code、Cursor、Codex 等 75+ harness。
- **为什么值得保留：** 本库大量实体页（mattpocock、Anthropic、Addy Osmani、GSAP 等）均经此 CLI 分发；`find-skills` 元技能指导代理按安装量与来源声誉检索 skills.sh，是 **技能发现层** 的官方入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/find-skills-skill.md`](../wiki/entities/find-skills-skill.md)

## README 要点（归纳）

- **安装技能：** `npx skills add owner/repo`、深链到子目录、`skills use owner/repo@skill --agent cursor` 临时生成 prompt。
- **检索：** `npx skills find [query] [--owner]`；公开排行榜 https://skills.sh/
- **捆绑元技能：** 仓库内 `skills/find-skills/` — 当用户问「有没有 skill 能做 X」时加载，流程含 leaderboard → find → 质量门槛（安装量/来源/stars）→ 可选 `npx skills add -g -y`。
- **与 agent-skills 关系：** README 示例常用 `vercel-labs/agent-skills`；本仓是 **安装器 + 发现**，非技能内容全集。

## 关联资料

- skills.sh 页归档：[`sources/sites/skills-sh-find-skills.md`](../sites/skills-sh-find-skills.md)
- 浏览器自动化对照：[`vercel-labs-agent-browser.md`](vercel-labs-agent-browser.md)
- 分发消费方示例：[mattpocock-skills.md](mattpocock-skills.md)、[addyosmani-agent-skills.md](addyosmani-agent-skills.md)
