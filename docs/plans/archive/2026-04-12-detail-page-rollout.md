# Robotics_Notebooks Detail Page Rollout Plan

> **For Hermes:** 先做能上线的 metadata-first detail page，再决定是否做 markdown 正文同步；不要一开始引入复杂前端框架。

**Goal:** 把当前 `site-data-v1.json` 里的 `detail_pages` 从“预览数据”推进成真正可访问的 `detail.html?id=...` 页面，并保证 GitHub Pages 上可正常读取导出数据。

**Architecture:** 第一阶段采用 **metadata-first** 方案：详情页直接消费 `docs/exports/site-data-v1.json`，渲染标题、摘要、标签、关联项、来源链接和回源路径，不依赖 markdown 正文直出。同步修正导出发布链路，让 `exports/` 数据跟着 `docs/` 一起部署到 GitHub Pages。

**Tech Stack:** 原生 HTML / CSS / JS、Python 导出脚本、GitHub Pages（`docs/` 目录部署）、`unittest`

---

## 0. 当前阶段判断

### 已有基础
- `scripts/export_minimal.py` 已生成 `exports/index-v1.json` 和 `exports/site-data-v1.json`
- `docs/site-data-preview.html` 已验证 `detail_pages` / `tech_map_page` 数据存在
- `docs/main.js` 已有预览渲染逻辑

### 当前关键缺口
1. 还没有真正的 `docs/detail.html`
2. `main.js` 还没有详情页渲染器
3. GitHub Pages 只部署 `docs/`，但导出 JSON 目前在仓库根目录 `exports/`，**部署后页面拿不到数据**
4. tech-map / preview 还不能跳到真正的 detail page

### 实施原则
- 先解决“能访问、能跳转、能消费数据”
- 先渲染元信息，不急着做 markdown 正文渲染
- 保持单文件前端，不引入打包器

---

## 1. 本次拆分后的具体执行顺序

### Task 1: 先写失败测试，锁定 detail page 骨架
**Objective:** 用最小结构测试锁定 `detail.html` 和 `main.js` 的接入点。

**Files:**
- Create: `tests/test_detail_page.py`

**验证点：**
- `docs/detail.html` 存在
- 页面包含关键挂载点：breadcrumb / title / summary / meta / tags / related / sources / empty state
- `docs/main.js` 存在 `renderDetailPage()` 和 `URLSearchParams`

**Run:**
- `python -m unittest tests/test_detail_page.py`

---

### Task 2: 修正导出发布链路
**Objective:** 保证 GitHub Pages 上真的能拿到 `site-data-v1.json`。

**Files:**
- Modify: `scripts/export_minimal.py`
- Generate: `docs/exports/index-v1.json`
- Generate: `docs/exports/site-data-v1.json`

**做法：**
- 导出脚本在写根目录 `exports/` 的同时，同步写 `docs/exports/`
- 页面统一从 `exports/site-data-v1.json` 读取，不再使用 `../exports/...`

**验证：**
- 本地存在 `docs/exports/site-data-v1.json`
- `docs/site-data-preview.html` 和 `docs/detail.html` 都指向 `exports/site-data-v1.json`

---

### Task 3: 落地 metadata-first detail page
**Objective:** 让任意 detail id 都能渲染成真实页面。

**Files:**
- Create: `docs/detail.html`
- Modify: `docs/main.js`
- Modify: `docs/style.css`

**页面最低展示内容：**
- 标题
- 类型 / path / status
- summary
- tags
- related items（跳 `detail.html?id=...`）
- source links（外链）
- 空态 / 无效 id 提示

**URL 约定：**
- `detail.html?id=wiki-concepts-centroidal-dynamics`
- `detail.html?id=entity-isaac-gym-isaac-lab`
- `detail.html?id=tech-node-control-mpc`

---

### Task 4: 把 preview / tech-map 接到 detail page
**Objective:** 不只是有 detail page，还要让现有预览流真正能跳过去。

**Files:**
- Modify: `docs/main.js`

**做法：**
- `detailPreviewGrid` 的卡片加“打开详情页”按钮
- tech-map 节点卡片标题或按钮跳 `detail.html?id=...`
- related items 渲染为内部链接

---

### Task 5: 验证闭环
**Objective:** 确认最小 detail page 在结构和数据层都成立。

**Files:**
- Run only

**验证命令：**
- `python -m unittest tests/test_detail_page.py`
- `python scripts/export_minimal.py`
- 可选人工验证：打开 `docs/detail.html?id=wiki-concepts-centroidal-dynamics`

**验收标准：**
- 测试通过
- `docs/exports/site-data-v1.json` 已生成
- detail 页面能显示 title / summary / tags / related / sources
- 无效 id 会显示空态，不白屏

---

## 2. 这次先不做什么

### 暂不做 markdown 正文渲染
原因：
- GitHub Pages 当前只发 `docs/`
- markdown 正文在仓库其他目录，不直接可用
- 如果现在强行做，会引入额外同步链路或 markdown parser

### 暂不做多页面 router
原因：
- `detail.html?id=...` 已够验证第一阶段页面消费模型
- 当前重点是验证 schema 是否足够支撑真实页面

### 暂不做搜索框
原因：
- 现在先把“可访问详情页”打通
- 搜索属于第二阶段入口优化

---

## 3. 下一阶段候选

当 metadata-first 版本稳定后，再二选一：

### 方向 A：正文同步版 detail page
- 导出脚本把 markdown 正文或 HTML 摘要同步到 `docs/generated/`
- detail 页展示正文节选或全文

### 方向 B：tech-map 真正数据驱动化
- 把当前静态 `tech-map.html` 改成真正消费 `tech_map_page`
- 点击节点统一进入 detail page

我建议优先顺序：
1. 本次先完成 metadata-first detail page
2. 再做 tech-map data-driven
3. 最后决定是否做 markdown 正文同步
