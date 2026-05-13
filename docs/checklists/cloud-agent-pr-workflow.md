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

在将 PR 交给人类 review 前，建议附上**可读的验证证据**。默认应包含 **静态站点上、与本次改动直接对应的页面截图**（读者在 GitHub Pages 上会看到的同一套 `docs/` 渲染），而不是 GitHub PR 列表页或 diff 页的截图。

### 4.1 站点详情页（wiki / entity 正文）

知识页、实体页在站点上由 **`docs/detail.html`** 渲染，数据来自 `docs/exports/site-data-v1.json`（由 `make ci-preflight` 生成）。截图前应确保本地 `docs/exports/` 与当前分支一致。

1. **启动本地静态服务**（在仓库根目录）  
   ```bash
   cd docs && python3 -m http.server 8765
   ```
2. **构造 URL**  
   - 形式：`http://127.0.0.1:8765/detail.html?id=<detail-page-id>`  
   - **`<detail-page-id>`** 与导出一致：可查 `docs/sitemap.xml` 中的 `detail.html?id=...`，或在 `site-data-v1.json` 的 `pages.detail_pages` 下找键名（例：`wiki-concepts-humanoid-parallel-joint-kinematics`）。  
3. **Headless Chrome 截图**（示例）  
   ```bash
   google-chrome --headless=new --disable-gpu --no-sandbox \
     --user-data-dir=/tmp/chrome-headless-screenshot \
     --window-size=1280,2200 --virtual-time-budget=12000 \
     --screenshot=/path/to/site-detail-<topic>.png \
     "http://127.0.0.1:8765/detail.html?id=wiki-concepts-..."
   ```  
   若外网字体/CSS 加载较慢，可加 `--disable-remote-fonts` 或适当增大 `--virtual-time-budget`。  
4. **已合并且 Pages 已更新时**（可选）  
   亦可使用线上等价 URL，例如：  
   `https://imchong.github.io/Robotics_Notebooks/docs/detail.html?id=...`  
   注意：功能分支在合并前通常**不会**出现在线上，此时必须以本地 `http.server` 为准。

### 4.2 命令行门禁摘要（补充，不能替代站点页）

仍建议在 PR 中用文字或**额外**截图说明 `make ci-preflight` / `make ci-check` / `make lint` 等已通过（可将命令输出写入本地 HTML 再 headless 截图）。**不应**用「仅 PR 页面截图」代替站点详情页验证。

### 4.3 嵌入 PR 描述

- 在 PR 正文中增加 **「验证截图」** 小节，使用 HTML：`<img alt="..." src="<绝对路径>" />`。  
- Cloud 环境会将此类绝对路径下的图片**上传并重写为稳定 URL**，因此**无需**把截图二进制提交进 Git 历史（见下方路径约定）。  

### 截图输出路径约定

- 优先写入可写目录：`<workspace>/.cursor-artifacts/screenshots/`（本仓库 `.gitignore` 已忽略该目录）。  
- 若运行环境允许写入 `/opt/cursor/artifacts/screenshots/`，亦可使用（与部分内部工具文档中的示例一致）。  

### 4.4 Playwright 兜底方案（当 `google-chrome` 不可用时）

某些 Cloud 容器没有预装完整桌面库，Playwright 可能报错：`error while loading shared libraries: libatk-1.0.so.0`。可先补系统依赖，再截图：

```bash
sudo apt-get update
sudo apt-get install -y \
  libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
  libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2t64
```

然后用固定版本执行（避免 `npx playwright` 每次临时安装导致浏览器缓存不一致）：

```bash
npx -y playwright@1.60.0 install chromium
npx -y playwright@1.60.0 screenshot --device='Desktop Chrome' \
  'http://127.0.0.1:8765/roadmap.html?id=roadmap-motion-control' \
  '<workspace>/.cursor-artifacts/screenshots/roadmap-motion-control-fix.png'
```

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
