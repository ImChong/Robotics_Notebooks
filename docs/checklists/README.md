# 执行清单索引

`docs/checklists/` 保存项目阶段性执行计划、验收标准和历史推进记录。它不是 wiki 知识层，而是维护者看板。

## 当前入口

- [技术栈项目执行清单 v25](tech-stack-next-phase-checklist-v25.md) — 当前技术栈、自动化、专题建设与 UX 推进看板。
- [前端体验优化清单 v1](frontend-optimization-v1.md) — GitHub Pages 首页与交互体验优化计划。
- [Cursor Cloud Agent：PR 与验证截图流程](cloud-agent-pr-workflow.md) — Cloud Agent 推送分支、开 PR、附验证截图的路径约定。
- [GitHub Actions CI 门禁](github-actions-ci-gate.md) — 合并 `main` 前必须等全量 Actions 全绿；branch protection 建议。

## 历史执行清单

这些文件记录从 v1 到 v23 的阶段性目标和已完成事项，主要用于追溯决策，不作为当前待办入口。历史版本统一存放在 [`archive/`](archive/) 子目录。

- [v1](archive/tech-stack-next-phase-checklist-v1.md)
- [v2](archive/tech-stack-next-phase-checklist-v2.md)
- [v3](archive/tech-stack-next-phase-checklist-v3.md)
- [v4](archive/tech-stack-next-phase-checklist-v4.md)
- [v5](archive/tech-stack-next-phase-checklist-v5.md)
- [v6](archive/tech-stack-next-phase-checklist-v6.md)
- [v7](archive/tech-stack-next-phase-checklist-v7.md)
- [v8](archive/tech-stack-next-phase-checklist-v8.md)
- [v9](archive/tech-stack-next-phase-checklist-v9.md)
- [v10](archive/tech-stack-next-phase-checklist-v10.md)
- [v11](archive/tech-stack-next-phase-checklist-v11.md)
- [v12](archive/tech-stack-next-phase-checklist-v12.md)
- [v13](archive/tech-stack-next-phase-checklist-v13.md)
- [v14](archive/tech-stack-next-phase-checklist-v14.md)
- [v15](archive/tech-stack-next-phase-checklist-v15.md)
- [v16](archive/tech-stack-next-phase-checklist-v16.md)
- [v17](archive/tech-stack-next-phase-checklist-v17.md)
- [v18](archive/tech-stack-next-phase-checklist-v18.md)
- [v19](archive/tech-stack-next-phase-checklist-v19.md)
- [v20](archive/tech-stack-next-phase-checklist-v20.md)
- [v21](archive/tech-stack-next-phase-checklist-v21.md)
- [v22](archive/tech-stack-next-phase-checklist-v22.md)
- [v23](archive/tech-stack-next-phase-checklist-v23.md)
- [v24](archive/tech-stack-next-phase-checklist-v24.md)

## 维护规则

- 开启新阶段时，新建 `tech-stack-next-phase-checklist-v{N}.md`，把上一版移入 `archive/`，并在本索引的“当前入口”和“历史执行清单”中更新链接。
- 前端体验专项任务优先写入 `frontend-optimization-v1.md`；若后续拆出 v2，应保留 v1 作为历史记录。
- checklist 完成项要及时从 `[ ]` 更新为 `[x]`，并在重要节点同步记录到根目录 `log.md`。
