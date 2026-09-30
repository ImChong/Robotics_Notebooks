# Claude Code Game Studios（Donchitos/Claude-Code-Game-Studios）

> 来源归档（ingest）

- **标题：** Claude Code Game Studios（CCGS）
- **类型：** repo / claude-code / skills / multi-agent / game-dev-template
- **作者 / 组织：** Donchitos（[GitHub @Donchitos](https://github.com/Donchitos)）
- **代码：** <https://github.com/Donchitos/Claude-Code-Game-Studios>（**已开源**，MIT）
- **项目页：** 无独立站点；以 GitHub README 与 [Claude Code 文档](https://code.claude.com/docs) 为准
- **许可：** MIT（以仓库 `LICENSE` 为准）
- **入库日期：** 2026-09-30
- **一句话说明：** 把单次 **Claude Code** 会话结构化成「虚拟游戏工作室」：**49 个子代理**、**74 个 slash skills**、**12 个 hooks**、**13 条路径规则** 与 **39 个文档模板**；通过 `project.yaml` 的 `modes.rigor`（minimal / standard / full）统一调节流程重量，默认强调 **人决策、代理协作而非自动驾驶**。
- **为什么值得保留：** 与本库 [Superpowers](../../wiki/entities/superpowers-obra.md)、[CLI-Anything](../../wiki/entities/cli-anything.md)（含 Godot harness）、[image-blaster](../../wiki/entities/image-blaster.md) 同属 **Claude Code 技能 + 多代理编排** 谱系；对 **仿真/交互 3D 原型**（Godot / Unity / UE5 引擎专家代理集）与 **agentic 交付流程**（GDD→故事→实现→QA 证据）有选型对照价值。

## 开源状态（步骤 2.5）

| 项 | 核查（2026-09-30） |
|----|-------------------|
| **GitHub** | 公开仓 [Donchitos/Claude-Code-Game-Studios](https://github.com/Donchitos/Claude-Code-Game-Studios)；默认分支 `main`；MIT |
| **独立项目页** | 无；README 链到 Claude Code 官方文档与 Buy Me a Coffee / Sponsors |
| **结论** | **已开源**（完整 `.claude/` 代理、技能、hooks、规则与模板；克隆即用，非权重/数据集类资源） |

## README / 架构要点（归纳，2026-09-30）

- **定位：** 单人 + AI 做游戏时，补齐「设计评审、QA、文档、域边界」等工作室结构；**不是**自动写完整个游戏的 autopilot。
- **规模（README 徽章，以仓内为准）：** 49 agents、74 skills、12 hooks、13 rules、39 templates。
- **层级：** Tier1 总监（creative / technical / producer）→ Tier2 部门负责人 → Tier3 专家（含 Godot / Unity / Unreal 引擎 specialist 子集）。
- **配置中枢：** 根目录 `project.yaml`；`modes.rigor` 为总门（minimal 默认；standard / full 提高 GDD、QA 证据与 director review）；`system_overrides` 可对单系统提标；个人覆盖写 gitignore 的 `project.local.yaml`。
- **实证叙事（README）：** 同一 brief 在四种 rigor 下测「首行游戏代码前文档数」；`standard` 档文档多但盲评游玩质量未必更好；**视觉/UI 改动** 故事关闭前须 **启动游戏、观察并保留截图** 于 `production/qa/evidence/`（各 rigor 均适用）。
- **协作协议：** Ask → 多选项 → 用户决策 → Draft → Approve；垂直委派、水平咨询、冲突升级至总监/ producer。
- **安全自动化：** `settings.json` hooks（commit/push/asset/skill 变更校验、session 审计、agent spawn 日志等）+ 权限规则（阻断 force push、`rm -rf`、读 `.env` 等）。
- **自测框架：** `CCGS Skill Testing Framework/` + `/skill-test`、`/skill-improve` 用于验证 **框架技能/代理** 编辑，与 `tests/` 内 **游戏** 测试分离。
- **前置依赖：** Git、Claude Code、Python 3（读 `project.yaml`）、Bash；推荐 jq。

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/claude-code-game-studios.md`](../../wiki/entities/claude-code-game-studios.md)**
- 交叉更新 Superpowers、CLI-Anything、image-blaster、Agentic Coding 概念页「关联页面」
