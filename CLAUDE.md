# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. Claude Code Agent：PR 与验证截图

与 [Cursor Cloud Agent 流程](docs/checklists/cloud-agent-pr-workflow.md) 对齐：当本仓库由 **Claude Code Agent**（含 web / CLI / GitHub Action 任一形态）完成改动并推 PR 时，**默认应在 PR 正文中附验证截图**，证据级别按下表：

| 改动类型 | 默认必附 |
|----------|----------|
| 新增 / 改写 wiki 页面 | 该页面在静态站 `docs/detail.html?id=<page-id>` 上的渲染截图 |
| 新增 / 改写 roadmap 页面 | 对应 `docs/roadmap.html?id=...` 渲染截图（含 Mermaid 落稳） |
| 仅改 schema / 维护工具 / 脚本 | `make ci-preflight`（或等价门禁）输出截图即可 |
| 文档（如本文件本身）非渲染类改动 | 文字说明 + 命令输出即可，不强制截图 |

### 5.1 三步走（与 Cursor Cloud Agent 同流程）

1. **本地起静态服务**（仓库根目录）：
   ```bash
   cd docs && python3 -m http.server 8765
   ```
2. **构造 detail-page-id 并截图**（推荐使用仓库脚本，自带 timeout + PNG 落盘判据）：
   ```bash
   ./scripts/screenshot_site_detail.sh wiki-entities-paper-behavior-foundation-model-humanoid
   # 自定义输出路径：
   ./scripts/screenshot_site_detail.sh wiki-entities-paper-... .cursor-artifacts/screenshots/bfm.png
   # 截局部（如 detail-sources 区块）：
   ./scripts/screenshot_site_detail.sh wiki-entities-paper-... out.png detail-sources
   ```
   - `<page-id>` 可在 `docs/sitemap.xml` 或 `docs/exports/site-data-v1.json` 的 `pages.detail_pages` 找到。
   - Cloud 环境没有 Chrome 时先 `sudo -n apt-get install -y -f` 装齐依赖，或按 [cloud-agent-pr-workflow.md §4.4](docs/checklists/cloud-agent-pr-workflow.md) 用 Playwright 兜底。
3. **嵌入 PR 描述**：在「验证截图」小节用 HTML 引用绝对路径，路径默认落 `.cursor-artifacts/screenshots/`（已被 `.gitignore` 屏蔽，不要提交进 git 历史）。
   ```html
   <img alt="BFM 详情页截图" src="/home/.../.cursor-artifacts/screenshots/bfm.png" />
   ```
   Cloud 环境会自动把绝对路径下的图片上传并重写为稳定 URL。

### 5.2 PR 正文最小骨架

```markdown
## 摘要
- <一句话改了什么 / 为什么>

## 验证
- `make ci-preflight` 通过（或：N/A，仅非 wiki 改动）
- 静态站详情页渲染正常

## 验证截图
<img alt="..." src=".cursor-artifacts/screenshots/<page>.png" />
```

### 5.3 与 AGENTS.md / cloud-agent-pr-workflow.md 的关系

- **流程主文档**：[docs/checklists/cloud-agent-pr-workflow.md](docs/checklists/cloud-agent-pr-workflow.md) — 分支、`ci-preflight`、截图脚本细节、Playwright 兜底等。
- **AGENTS.md** 内的「Cursor Cloud specific instructions」给出了同一指针；CLAUDE.md 这一节是 **Claude Code Agent 视角的对齐说明**，避免每次重新查阅。
- 中文 commit 规范（ingest / structural / fix 等格式）见 [AGENTS.md](AGENTS.md) §「Git 提交规范」。
