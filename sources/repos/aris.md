# ARIS — Auto-Research-In-Sleep（官方原始仓库）

- **类型：** repo / 自主科研 Skills 及协作脚手架
- **原始仓库：** https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep
- **上游身份：** GitHub REST `fork: false`
- **核验版本：** `8f3110718259`（main，2026-11-08）
- **入库日期：** 2026-11-11
- **许可：** MIT
- **开源状态：** `skills/`、`mcp-servers/`、`tools/`、`docs/` 与实验编排相关文件已公开
- **核对范围：** 已全文阅读 `docs/CUSTOMIZATION.md` 的配置和预算规则；核对 `AGENT_GUIDE.md`、`skills/research-pipeline/SKILL.md`、`skills/experiment-bridge/SKILL.md` 与 `skills/auto-review-loop/SKILL.md` 的入口和部分实现约束；主 README 长约 232k 字符，仅阅读前部章节，未完成通篇阅读，未启动 GPU 实验。
- **通读：** 未完成
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

## 待复核

主 README 尚未通读，自动化控制依赖宿主 Agent、远程计算资源及其权限；未提供经过本次独立复测的实验效果与预算保证。