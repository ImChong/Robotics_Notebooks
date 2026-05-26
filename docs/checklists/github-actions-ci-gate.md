# GitHub Actions CI 门禁（合并前必等）

> 维护者看板，非 wiki 知识页。用于在 PR #387 等「未触发 Actions 即合并」之后，统一 **合并前必须等 CI 全绿** 的约定。

## 目标

对指向 `main` 的 Pull Request，在 GitHub 上应出现并 **全部通过** 下列工作流（名称以 Actions 页为准）后，才允许合并：

| 工作流 | 触发（PR） | 大致耗时 |
|--------|------------|----------|
| **Tests** | 任意 PR → `main` | ~1 分钟 |
| **Wiki Lint** | 任意 PR → `main` | ~1 分钟 |
| **Search & Export Quality Check** | PR 改动 `wiki/`、`sources/`、`scripts/`、`schema/` 等 | ~7–10 分钟 |

合并进 `main` 后（push 到 `main`）还会跑：

| 工作流 | 说明 |
|--------|------|
| **Auto Export & Lint** | `wiki/` / `sources/` 等变更时同步派生文件 |
| **Deploy GitHub Pages** | 站点发布 |

本地 **`make ci-preflight`** 与 GitHub Actions **互补**：提交前本地预检；PR 上仍以 Actions 检查结果为准。

## 合并前检查清单

1. 打开 PR 页 **Checks** 标签，确认上述三项均为绿色（非「跳过 / 未运行」）。
2. 若 Checks 为空：到仓库 **Settings → Actions** 确认已启用；在 Actions 页对 `Tests` 等手动 **Run workflow** 排查。
3. 建议仓库管理员在 **Settings → Branches → Branch protection rules** 为 `main` 勾选 Required status checks（至少 `Tests`、`Wiki Lint`；ingest 类 PR 再加 `Search & Export Quality Check`）。
4. Cloud Agent：**不要**在 Draft 刚转 Ready 后数分钟内合并；`Search & Export Quality Check` 常需 7 分钟以上。

## 与本仓库其它文档

- 本地命令与派生文件同步：[contributing-ci.md](../contributing-ci.md)
- Cloud Agent 开 PR / 截图：[cloud-agent-pr-workflow.md](cloud-agent-pr-workflow.md)
- 根目录维护说明：[AGENTS.md](../../AGENTS.md)（Cursor Cloud / `make ci-preflight`）

## 参考

- 工作流定义：`.github/workflows/tests.yml`、`lint.yml`、`search-regression.yml`、`export.yml`
- 问题背景：PR #387 合并时 `statusCheckRollup` 为空、该时段仓库无新 Actions run 记录（2026-05-26）
