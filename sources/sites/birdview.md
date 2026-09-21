# Birdview 项目页（qiuner.github.io/birdview）

> 来源归档

- **标题：** Birdview — See what AI will change
- **类型：** site / project-page
- **URL：** <https://qiuner.github.io/birdview/>
- **代码：** <https://github.com/Qiuner/birdview> — [`sources/repos/birdview.md`](../repos/birdview.md)
- **作者：** Qiuner
- **入库日期：** 2026-09-21
- **一句话说明：** 官方静态站：Architecture-first Coding 叙事、Codex / Claude Code 等安装命令、`doctor` 自检与 QQ 用户群入口。

## 开源核查（步骤 2.5，截至 2026-09-21）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **是** — Install 区与页脚指向 `github.com/Qiuner/birdview` |
| 训练/推理入口 | **不适用**（开发者工具 Skill，非 ML 框架） |
| 可运行实现 | **有** — `npx skills add Qiuner/birdview --skill birdview`；仓内 `node scripts/birdview.mjs doctor`；示例 `examples/harness-activity.html` |
| 数据 / 权重 | **无** |
| npm 发布 | **未发布到 npm registry**（README 写明 package 当前 private；通过 skills CLI 从 GitHub 安装） |
| 综合判定 | **已开源**（MIT） |

## 页面要点（2026-09-21 抓取）

- Hero：**Stop letting AI code blind** — 强调改码前先 map；口号「constraints and architecture」。
- 三能力块：Architecture context（稳定模块身份 + 证据）、Change scope（计划模块高亮）、Verifiable output（JSON 校验 → 自包含 HTML）。
- 安装示例（Codex）：`npx skills add Qiuner/birdview --skill birdview --agent codex --global --copy --yes` → `npm --prefix "$HOME/.agents/skills/birdview" ci` → `node .../birdview.mjs doctor` → 新任务中显式调用 Skill。
- 社区：QQ 群 627760389；GitHub Issues usage feedback 模板。
- 模式说明：页内写默认 **auto** 与 on-demand 优先级以 README / `references/modes.md` 为准 — README 明确 **新项目默认 on-demand**，已有 `Birdview mode: auto` 块仍生效。

## 关联资料

- 仓库归档：[`sources/repos/birdview.md`](../repos/birdview.md)
- Wiki：[`wiki/entities/birdview.md`](../../wiki/entities/birdview.md)
