# Cursor Cloud Agent：推送 PR 与验证截图流程

> 本文件记录在本仓库内由 **Cloud Agent** 完成改动时的推荐收尾流程，供人类 curator 与后续 agent 对齐；**不属于** wiki 知识页。

## 1. 分支与提交

1. 自 `main`（或任务指定的 base）检出功能分支，名称使用仓库约定前缀与后缀（例如 `cursor/<topic>-e361`）。
2. 涉及 `wiki/` 或派生索引时，提交前**必须**运行 `make ci-preflight`（或任务要求的等价 CI 门禁），避免 `exports/`、`docs/search-index.json` 等与远端不一致。
3. 仅 `stage` 与本次任务相关的文件；提交信息遵循根目录 [`AGENTS.md`](../../AGENTS.md) 中的 **中文 commit 规范**。

## 2. 推送远端

```bash
git push -u origin <branch-name>
```

推送失败时按网络情况退避重试（见 Cloud 任务说明）。

## 3. 创建或更新 Pull Request

- 使用仓库的 PR 管理流程创建草稿或正式 PR，`base_branch` 与任务要求一致（未指定时默认 `main`）。
- PR 正文应包含：**摘要**（改了什么、为什么）、**风险或回滚注意**（如有）、**关联 issue**（如有）。

## 4. 验证截图（推荐默认执行）

在将 PR 交给人类 review 前，建议附上**可读的验证证据**，避免「只声称跑过 CI」：

1. **本地门禁摘要**（至少其一，按改动类型选）  
   - Wiki / 导出相关：`make ci-preflight` 或 `make ci-check` 成功输出的**末尾若干行**。  
   - 纯脚本：`make test` 或 `make ci-test` 中与本次改动相关的部分。  
2. **生成截图的实用做法**  
   - 将验证输出写入本地 HTML（含通过标识与 `git log -1 --oneline`），用 **Headless Chrome** 对 `file://` 页面截图；或  
   - 对 GitHub 上本 PR 页面做 headless 截图（需网络；建议加 `--user-data-dir` 避免 profile 冲突）。  
3. **嵌入 PR 描述**  
   - 在 PR 正文中增加 **「验证截图」** 小节，使用 HTML：`<img alt="..." src="<绝对路径>" />`。  
   - Cloud 环境会将此类绝对路径下的图片**上传并重写为稳定 URL**，因此**无需**把截图二进制提交进 Git 历史（见下方路径约定）。  

### 截图输出路径约定

- 优先写入可写目录：`<workspace>/.cursor-artifacts/screenshots/`（本仓库 `.gitignore` 已忽略该目录）。  
- 若运行环境允许写入 `/opt/cursor/artifacts/screenshots/`，亦可使用（与部分内部工具文档中的示例一致）。  

## 5. 迭代中的 PR 更新

若验证或 CI 失败后修复再推送：

1. `git commit` + `git push` 更新同一分支。  
2. **再次**更新 PR 描述或验证截图（若结论有变化）。  
3. 人类合并后，可在本 checklist 或 `log.md` 中按需记一笔闭环说明（非强制）。

## 6. 与 AGENTS 的关系

根目录 [`AGENTS.md`](../../AGENTS.md) 的 **Cursor Cloud specific instructions** 中会保留对本文件的**简短指针**；详细步骤与路径约定以**本文件**为准，避免 AGENTS 过长。

## 参考来源

- 仓库根目录 [`AGENTS.md`](../../AGENTS.md) — LLM Wiki Ops 与 CI 门禁说明  
- [`schema/ingest-workflow.md`](../../schema/ingest-workflow.md) — ingest / 升格 wiki 的通用规范（与 PR 流程互补）
