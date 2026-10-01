# ECC（affaan-m/ECC）

> 来源归档

- **标题：** ECC — Everything Claude Code / Agent harness performance optimization
- **类型：** repo（多 harness 代理「工程操作系统」：skills + agents + hooks + memory + security）
- **作者：** affaan-m（社区维护；README 强调仅从官方渠道安装）
- **链接：** https://github.com/affaan-m/ECC
- **官网 / 产品：** https://ecc.tools — [`sources/sites/ecc-tools.md`](../sites/ecc-tools.md)
- **许可：** MIT（OSS 永久免费；ECC Pro GitHub App 为托管商业层）
- **入库日期：** 2026-10-01
- **Trendshift（用户触发，2026-10）：** 约 **+26.8k stars/月**；GitHub API 2026-10-01 约 **270k** stars
- **一句话说明：** 代理 **性能与工程纪律** 优化系统：一次安装接入 **plan → test → implement → review → verify → remember → improve** 闭环；含 **68 agents、293 skills、94 commands**、hooks/continuous learning/instincts、**AgentShield** 安全扫描；主推 Claude Code，Codex 有 sync path，Cursor/OpenCode/Gemini 等为 capability-limited adapters。
- **为什么值得保留：** 与 [Superpowers（obra）](../../wiki/entities/superpowers-obra.md)、[Agent Skills（Addy Osmani）](../../wiki/entities/agent-skills-addyosmani.md) 同属 **skills-first SDLC** 谱系，但 ECC 强调 **跨 harness 打包规模 + 记忆/安全/研究优先**；star 体量使其成为编码 Agent 生态的 **默认对照组** 之一。
- **沉淀到 wiki：** 是 → [`wiki/entities/ecc.md`](../../wiki/entities/ecc.md)

## README 要点（归纳，2026-10-01）

- **口号：** *Optimize the context window. Persist everything else.*
- **安装（推荐）：** `npx ecc-universal@2.2.2 setup`（Node ≥18）；Claude 插件 `ecc@ecc`；npm `ecc-universal`、`ecc-agentshield`。
- **官方渠道警告：** 仅 GitHub、ecc.tools、列名 npm 包、GitHub App `ecc-tools`；勿用未审核镜像。
- **平台：** 见 upstream `#platform-support` 矩阵 — Claude Code 功能最全。
- **商业：** ECC Pro + GitHub App（私有仓）；赞助与 Pro 资助开源维护。

## 开源状态

- **已开源（MIT）** — 主体仓库与 npm 包；Pro 为可选托管。

## 对 wiki 的映射

| 目标 | 链接 |
|------|------|
| 实体页 | [`wiki/entities/ecc.md`](../../wiki/entities/ecc.md) |
| 流程技能对照 | [`wiki/entities/superpowers-obra.md`](../../wiki/entities/superpowers-obra.md) |
| SDLC 技能包 | [`wiki/entities/agent-skills-addyosmani.md`](../../wiki/entities/agent-skills-addyosmani.md) |
