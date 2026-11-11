# ARIS — Auto-Research-In-Sleep（官方原始仓库）

- **类型：** repo / 自主科研 Skills 及协作脚手架
- **原始仓库：** https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep
- **上游身份：** GitHub REST `fork: false`
- **核验版本：** `8f3110718259`（main，2026-11-08）
- **入库日期：** 2026-11-11
- **许可：** MIT
- **开源状态：** `skills/`、`mcp-servers/`、`tools/`、`docs/` 与实验编排相关文件已公开
- **核对范围：** 已通读核心一手技术文档 `AGENT_GUIDE.md`、`skills/research-pipeline/SKILL.md`、`skills/run-experiment/SKILL.md` 和 `docs/CUSTOMIZATION.md`；通读主 README 的 Quick Start、Features、Workflows、Setup 等技术正文，历史 News/Release 巨量明细仅作目录与重点核对；精读 `skills/experiment-bridge/SKILL.md` 的实验实现、部署及汇总阶段，`skills/auto-review-loop/SKILL.md` 的 reviewer 路由和停止门禁；已逐段核验 `mcp-servers/codex-exec/server.py` 的 `main`、`handle_call`、`run_codex` 以及 thread 配置持久化。阅读范围覆盖本实体的主要机制与实现入口，不等于通读仓库内所有 Skill 或完成 GPU 运行测试。
- **通读：** 已完成
- **知识页：** [ARIS](../../wiki/entities/aris-auto-research-in-sleep.md)

## 官方原始资料

- [官方仓库 README](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/8f3110718259/README.md)
- [Agent 指引](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/8f3110718259/AGENT_GUIDE.md)
- [定制参数/预算](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/8f3110718259/docs/CUSTOMIZATION.md)
- [完整研究管线 Skill](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/8f3110718259/skills/research-pipeline/SKILL.md)
- [实验桥接 Skill](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/8f3110718259/skills/experiment-bridge/SKILL.md)
- [自动评审循环 Skill](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/8f3110718259/skills/auto-review-loop/SKILL.md)

## 已验证的参数边界（不是全局保证）

- `idea-discovery`/Pilot：`PILOT_MAX_HOURS=2`、`PILOT_TIMEOUT_HOURS=3`、`MAX_PILOT_IDEAS=3`、`MAX_TOTAL_GPU_HOURS=8`，只覆盖该阶段的候选实验。
- `experiment-bridge`：`CODE_REVIEW=true`、`AUTO_DEPLOY=true`、`SANITY_FIRST=true`、`MAX_PARALLEL_RUNS=4`；并行数限制不等于全局 GPU 时长或费用限制。
- `auto-review-loop`：`MAX_ROUNDS=4`；实际终止还需满足审稿 verdict 条件，不能只看数值分数。
- 建议过夜任务将 `AUTO_DEPLOY=false` 并设外部 GPU 时长/费用上限、实验最大迭代次数、资源异常停止条件，确认任务边界后再授权运行。

## 源码调用链及复核边界

基于 [research-pipeline](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/8f3110718259/skills/research-pipeline/SKILL.md) 与 [experiment-bridge](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/8f3110718259/skills/experiment-bridge/SKILL.md)：

1. 宿主 Agent 消费 `research-pipeline`：`idea-discovery` 产出计划，`experiment-bridge` 核查代码、先运行 sanity，再根据任务数量按 `run-experiment` / `experiment-queue` 分流。
2. `run-experiment` 从项目 `CLAUDE.md` 判定本地 GPU / 远程 SSH / Vast.ai / Modal 目标；按需要同步代码，通过 shell `python <script> ... | tee <log>` 或 Modal launcher 启动，后用 `screen -ls`、本地进程或 Modal 日志确认启动，并收集实验日志。
3. `auto-review-loop` 的 reviewer 在默认 Codex MCP 路线调用 `mcp__codex__codex`；`mcp-servers/codex-exec/server.py` 的 `handle_call` 生成 `codex exec --json` 命令，`run_codex` 解析 `turn.completed` 和 agent message，并保存 `threadId` 的模型/配置/工作路径；后续 `codex-reply` 经 `codex exec resume` 接续。
4. `AUTO_WRITE=false` 为 `research-pipeline` 默认值，故主流程通常停在 `NARRATIVE_REPORT.md`，并非自动投稿或自动论文写作。审稿循环至多四轮，提前停止需 reviewer `score >= 6` **且** verdict 属于 `ready/almost`；还要遵守 reviewer 身份与独立性门禁。

**仍未验证：** 实机 GPU/云费用控制、所有 87 个技能的实现和全量 release 细目。上述一手核心技术文档已完成阅读核查，但尚不能据此承诺实验成功率或自动化资源上限。