# Robotics_Notebooks Detail Page Content Sync Plan

> **For Hermes:** 在已经跑通 metadata-first detail page 的基础上，先做最小正文同步，再把 raw markdown 升级成基础可读版渲染；仍然不引入前端构建链路和外部依赖。

**Goal:** 让 `detail.html?id=...` 在标题、摘要、标签之外，还能展示来自源 markdown 的基础可读正文，完成 detail page 从 metadata-only 到 content-backed 的第一步。

**Architecture:** 继续沿用 `site-data-v1.json`。导出层保持 `detail_pages[*].content_markdown`，内容为去掉一级标题后的 markdown 正文。前端 detail page 新增最小 markdown renderer，直接在浏览器里把标题、列表、引用、代码块、粗体、行内代码与链接渲染成基础 HTML；同时为二级到四级标题生成锚点，并自动生成最小 TOC。对于公式，先在渲染阶段识别 `\\(...\\)` 与 `$$...$$`，再通过 KaTeX auto-render 做真正数学排版。整个方案仍然不引入前端构建链路，只增加浏览器侧 CDN 依赖。

**Tech Stack:** Python 导出脚本、原生 HTML/CSS/JS、`unittest`

---

## 为什么先这样做

### 目标不是最终排版，而是先打通链路
当前真正关键的是：
1. 导出层是否能稳定携带正文
2. GitHub Pages 部署后页面能否读到正文
3. detail page 是否已经具备从 metadata-first 向 content-backed 演进的最小接口

### 为什么不用 markdown 渲染器
- 当前仓库没有前端打包链路
- 不想为了一个页面引入额外依赖
- 先验证“内容同步”比“漂亮渲染”更重要

---

## 执行步骤

### Task 1: 先写失败测试
文件：`tests/test_content_sync.py`

验证：
- `scripts/export_minimal.py` 里存在 `content_markdown`
- `docs/detail.html` 里存在：
  - `id="detailContent"`
  - `id="detailContentSection"`
- `docs/main.js` 里存在：
  - `document.getElementById('detailContent')`
  - `content_markdown`

### Task 2: 扩展导出脚本
目标：
- 为每个 `detail_page` 增加 `content_markdown`
- 内容取自 markdown 正文，去掉第一行 `# 标题`
- tech-map 节点如果没有 markdown 正文，也允许为空字符串

### Task 3: 扩展 detail page
目标：
- 新增“正文同步内容”分区
- 如果 `content_markdown` 有内容，就展示
- 如果没有，就隐藏该分区或显示轻提示

### Task 4: 扩展前端渲染器
目标：
- detail page 读取 `content_markdown`
- 用仓库内原生 JS 实现最小 markdown 渲染，不引入前端构建链路
- 第一阶段至少支持：标题、列表、引用、代码块、粗体、行内代码、链接
- 对 `\\(...\\)` 与 `$$...$$` 先做结构识别，再接入 KaTeX auto-render 进行真正数学排版
- 为长文自动生成最小 TOC 与标题锚点
- 继续补 TOC 当前阅读位置高亮，以及标题一键复制锚点
- 保持空态处理

### Task 5: 验证
命令：
- `python -m unittest tests/test_content_sync.py tests/test_detail_page.py`
- `python scripts/export_minimal.py`
- 浏览器打开 `detail.html?id=wiki-concepts-centroidal-dynamics`

验收：
- detail page 出现“正文同步内容”分区
- 页面能把来自 markdown 的正文渲染成基础可读结构
- 公式块与行内公式会被 KaTeX 真正排版，不再只是独立样式高亮
- 长文页面会自动出现目录导航，并能跳到对应标题
- TOC 会随当前阅读位置高亮当前章节，标题支持一键复制锚点
- tech-map 节点页若无正文，不会报错或白屏
