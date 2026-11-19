# 右下角只读 AI 知识库问答设计方案（Issue #3908）

> 日期：2026-11-19 · 关联 Issue：[#3908](https://github.com/ImChong/Robotics_Notebooks/issues/3908)
> 目标：**站长新增 AI 服务 0 元支出** · **多层防刷与配额硬停止** · **无工具/无写凭证强制隔离** · **统一右下角控件栈不遮挡**

---

## 一、Cloudflare Free 官方配额与约束复核（2026-11-19）

在动手设计前，已按官方文档复核 Cloudflare 免费套餐边界与模型生命周期：

| 组件 | 免费层官方配额 / 限制 | 对架构设计的约束 |
|---|---|---|
| **Workers Free** | **100,000 请求/天**；单次调用 **CPU 时间硬上限 10 ms**（不含 `fetch` / AI 等待的墙钟时间）；单 Isolate 内存 128 MB；外部子请求上限 50 次 | Worker 内 **严禁** 下载或解析 `site-catalog-v1.json`（~9 MB）或 `search-index.json`（~106 MB），否则必然触发 `1102 CPU time exceeded`。必须由前端检索定位候选节点，Worker 仅拉取至多 4 个 KB 级的 `page-content/<sha256>.json` 分片。 |
| **Workers AI Free** | **10,000 Neurons/天**（UTC 00:00 重置）；**保持 Workers Free 不升级 Paid 时触发硬停止（Hard Stop）**——耗尽直接报错，**不会**转入 `\$0.011 / 1,000 Neurons` 超额计费 | 「0 元支出」的最终物理保障是账户停留在 **Workers Free**、不绑定自动计费的第三方 Key、不启用 AI Gateway 付费兜底路由；Worker 捕获 AI 报错后 fail closed，前端自动回退本地只读检索。 |
| **Workers Rate Limiting Binding** | Workers Free 原生可用、无额外收费；支持 `period: 10` 或 `60` 秒；按边缘节点机房（colo）近似计数 | 用于作为网关第一道防刷闸门（先限速与并发锁，再进 Turnstile 与 AI），因分布式最终一致性，**不**作为全局精准计费保障。 |
| **Cloudflare Turnstile Free** | 免费无限次验证；单账户最多 20 个 Widget、每 Widget 最多 10 个域名；Token **单次有效且 300 秒过期**（≤ 2048 字符） | 前端采用按需显式渲染（`interaction-only`），未点击聊天入口时不请求 Turnstile；每次提交问答均附带一次性 Token，由 Worker 调用 `siteverify` 服务端强校验并核对 `hostname`。 |

### 模型弃用与 Neuron 预算核算

- **弃用避险**：核查 Cloudflare Workers AI 2026 年弃用清单，包括 `@cf/qwen/qwen3-30b-a3b-fp8` 在内的 24 个旧模型定于 **2026-11-20** 下线。
- **默认现役模型**：`@cf/meta/llama-3.1-8b-instruct-fp8-fast`
  - 官方折算：**4,119 Neurons / 百万输入 tokens**、**34,868 Neurons / 百万输出 tokens**。
  - 单次问答估算：输入（系统提示词 + ≤2,400 字符检索段落 + ≤300 字符提问）约 1,000 tokens（≈ 4.1 Neurons）+ 输出上限 `max_tokens: 384`（≈ 10–13 Neurons）≈ **15 Neurons/次**。
  - 免费额度吞吐：**10,000 Neurons/天 ≈ 550–650 次未缓存问答/天**；配合 Worker Edge Cache（对规范化公共问题缓存 6 小时），常见高频提问命中缓存后 **消耗 0 Neuron**。
- **模型白名单**：Worker 代码内置 `FREE_SAFE_MODELS` 严格白名单，禁止通过环境变量或请求参数切换至高消耗大模型或未核准模型；如未来 Cloudflare 调整模型目录，只需替换白名单内的现役轻量模型即可。

---

## 二、总体架构与零重复产物 RAG 链路

### 2.1 为什么不新增一套全库段落索引文件？

仓库现有 `scripts/export_minimal.py` 已在 GitHub Pages 部署阶段生成：
1. `docs/search-index-lite.json`（轻量元数据索引：标题、摘要、标签、别名）；
2. `docs/exports/site-catalog-v1.json`（含全站 `detail_pages` / `roadmap_pages` 到 `exports/page-content/<64hex>.json` 的内容寻址映射）；
3. `docs/exports/page-content/<64hex>.json`（每页独立 Markdown 正文 JSON，单页仅数 KB，SHA-256 不可变缓存）。

若额外导出一份全库切段大 JSON，会同步膨胀 Pages 构建产物（现已约 260+ MB）并增加部署耗时与客户端首包下载。因此本方案采用 **零重复产物复用**：

```
[浏览器（GitHub Pages 静态站）]
  ├─ 未打开聊天面板：0 额外脚本、0 索引下载、0 Turnstile 请求
  ├─ 点击右下角入口：懒加载 docs/kb-chat.js
  ├─ 首次提问：按需加载 search-index-lite.json（+ 必要时 site-catalog-v1.json）
  ├─ 本地召回：分词 + 同义词扩展 + 当前详情页/路线页上下文加权 → Top-4 候选节点
  ├─ 提取哈希：从 content_url（exports/page-content/<64hex>.json）提取 64 位 SHA-256
  │
  ├─▶ 分支 A（未配置云端或本地降级）：
  │     直接拉取同源 exports/page-content/<64hex>.json → 按 Markdown 标题切分段落并打分
  │     → 展示最相关段落摘要与可点击节点卡片（detail.html?id=... / roadmap.html?id=...）
  │
  └─▶ 分支 B（已配置 Worker URL + Turnstile Site Key）：
        向 Turnstile 申请一次性 Token → POST /chat { question, lang, turnstileToken, contentHashes }
             │
             ▼
[Cloudflare Workers Free 网关（workers/kb-chat/）]
  1. 严格限制请求体流字节数（<= 4096 bytes）与严格 JSON Schema（拒绝未知字段）
  2. 多层限速与并发锁（IPv4 / IPv6 /64 归一化）：边缘 Burst/Minute Limiter + Isolate 并发锁 + 单 IP 日限额
  3. Turnstile siteverify 服务端验证（SECRET 仅存于 Worker Secrets，核对 hostname）
  4. 边缘公共问答缓存（caches.default，命中直接返回，0 Neuron 消耗）
  5. 固定白名单拉取：仅拼接 https://imchong.github.io/Robotics_Notebooks/exports/page-content/<64hex>.json
     - 解析 Markdown 段落并做词法相关性打分；若无相关段落或 404 → 直接返回「未找到站内依据」（0 AI 调用）
  6. env.AI.run(MODEL_ID, { messages, max_tokens: 384, temperature: 0.1 }) —— 无 tools/functions
  7. 服务端强制绑定 citations（100% 来自刚读取的 page-content JSON），过滤回答文本中的原始 URL
```

---

## 三、交互设计与右下角统一浮动控件栈

### 3.1 避免与「回到顶部」及页脚冲突

原有 `.back-to-top`（`docs/style.css`）为独立 `position: fixed` 元素；若再挂第二个独立的 `position: fixed` 聊天按钮，在桌面端/移动端切换、滚动显示进度环以及靠近页脚触发 `--fab-lift` 时极易遮挡或抢点击。

改造方案（复用原有规则并保持零冲突）：
- **注入范围**：与 `.back-to-top` 一致，仅在存在 `body > .footer` 的页面注入（`index.html`、`detail.html`、`roadmap.html`、`hubs.html`、`company.html`、`lab.html`、`change-log.html`、`star-history.html`、`404.html`）；全屏画布页 `graph.html` 与跳转桩页 `references.html` 不注入。
- **统一纵向控件栈 `.fab-stack-right`**：
  - `position: fixed; bottom: calc(24px + var(--fab-lift, 0px)); right: calc(20px - (100vw - 100%));`（保留原 scrollbar-gutter 装订线补偿与 `--fab-lift` 页脚避让）；
  - 容器自身 `pointer-events: none`，仅内部可见子元素（`.back-to-top.is-visible`、`.kb-chat-toggle`、展开后的 `.kb-chat-panel`）设为 `pointer-events: auto`，绝不吞掉底层卡片链接点击。
- **底部横向展开与上方问答面板排布**：
  - 控件栈内分为上方的 `#kbChatSlot`（展开后的 `.kb-chat-panel`）与下方的横向按钮行 `.fab-stack-right__actions`（`[ #backToTop (48×48) ]` 在左、`[ #kbChatToggle (48×48) ]` 在右、折叠态合并入口 `[ #fabMainToggle (48×48) ]` 居中右对齐，`gap: 12px`）；`.kb-chat-slot` 设为 `display: contents`，收起态不占位；
  - 页面未滚动时（无需置顶），右下角仅显示 1 个 `#kbChatToggle` 按钮，点击直接打开问答面板；滚动出现阅读进度环后，两者合并收拢为 1 个 `#fabMainToggle`（外圈显示阅读进度环、内圈为 `↑ + 💬` 聚合图标）；点击 `#fabMainToggle` 后直接向左展开为「左 `#backToTop` + 右 `#kbChatToggle`」两个按钮（无多余关闭圆钮；点页面空白处或 `Esc` 收回为 1 个）；
  - 展开问答面板时，`.kb-chat-panel` 始终位于底部按钮行上方，宽度 `min(380px, calc(100vw - 32px))`，最大高度 `min(540px, calc(100dvh - 156px - var(--fab-lift, 0px)))`，滚动后展开的 `#backToTop` 始终停留在 `#kbChatToggle` 左侧；
  - 因此在 `320px` 窄屏下，底部左右仍留出足够间隙（左侧 `.sidebar-toggle` / `.home-nav-fab` 与右侧 `#backToTop` 相距 124px），右侧展开控件、面板与左下方导航按钮 **互不遮挡、零重叠**。

### 3.2 中英双语与无依据/降级提示

- **语言同步**：监听首页 `robotics-home-language`（`home-i18n.js` 更新 `document.documentElement.lang`）并提供面板内独立的 `中文 / EN` 一键切换，所有界面文案、提问提示词、模式徽标与错误/降级说明均完整支持中英双语。
- **来源溯源**：每条站内回答下方附结构化来源卡，标题与链接严格映射为 `detail.html?id=<id>` 或 `roadmap.html?id=<id>`，点击可直达对应知识节点。
- **无依据与配额降级**：无匹配段落时明确说明「知识库暂未收录直接依据」并引导关键词检索/图谱；云端未配置、限流（`429`）或 Neurons 额度耗尽（`503`）时，自动无缝回退至本地只读段落摘要检索，页面其余功能完全不受影响。

---

## 四、防刷额度与主机/仓库安全架构（OWASP LLM01 对齐）

### 4.1 防止匿名刷爆 AI 额度（多层纵深防御）

1. **边缘限流（先于 Turnstile 与 AI）**：
   - `CHAT_BURST_LIMITER`：10 秒最多 3 次；
   - `CHAT_MINUTE_LIMITER`：60 秒最多 8 次；
   - IPv6 按 `/64` 前缀归一化（`a:b:c:d::/64`），防止单台设备轮换后缀绕过限频。
2. **Isolate 内并发与单 IP 日限额**：
   - 每个归一化 IP 同时仅允许 **1 个在途推理请求**，防止并发请求穿透异步计数窗口；
   - 单个 Isolate 内维护有界（≤ 5,000 条）的单 IP UTC 日计数（默认上限 30 次/天）。
3. **服务端 Turnstile `siteverify`**：
   - 未通过 `POST https://challenges.cloudflare.com/turnstile/v0/siteverify` 或 `hostname` 不在白名单时直接返回 `403`。
4. **防「通用免费 LLM 代理」滥用**：
   - 不接受客户端传自定义上下文、系统提示词或任意文本；
   - Worker 仅从写死的 `PAGES_BASE_URL` 拉取经过 `^[a-f0-9]{64}$` 校验的 `page-content` 文件，并在调用 AI 前先做本地词项重合度校验——与知识库完全无关的闲聊/垃圾输入 **直接返回未找到（消耗 0 Neuron）**。
5. **公共问答结果缓存 + 输入/输出硬限制**：
   - `question` ≤ 300 字符、无多轮对话历史拼接、`max_tokens` ≤ 384、请求体 ≤ 4,096 字节；
   - 边缘 Cache 缓存公共只读问答 6 小时；遇到任何异常或限额均 **fail closed** 并由前端切换本地检索。

### 4.2 Prompt Injection、XSS 与凭证强隔离

- **非 Agent 模型**：`env.AI.run()` 不配置任何 `tools` / `functions` / MCP / 代码执行 / 文件系统 / Git / 外部网页抓取引擎。
- **零仓库与部署凭证**：Worker 与 GitHub 仓库之间零连接，不持有 `GITHUB_TOKEN`、PAT、Deploy Key 或 Cloudflare 部署凭证（GitHub Actions 中也不存 Cloudflare Secret）。
- **抗 SSRF 与路径穿越**：`contentHashes` 强制正则 `^[a-f0-9]{64}$`，URL 由常量前缀拼接，拒绝用户指定任何 URL 或相对路径。
- **前端零 `innerHTML` 模型输出渲染**：`docs/kb-chat.js` 渲染用户提问、AI 文本、段落摘要与来源标题 **全部使用 `document.createElement` + `textContent`**，任何 `<script>`、`<img onerror=...>`、`javascript:`、`data:` 均作为纯文本转义展示，引用链接白名单仅放行同源 `detail.html?id=...` 与 `roadmap.html?id=...`。
- **内容安全策略（CSP）**：
  - Worker API 返回 `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`、`X-Content-Type-Options: nosniff`、`X-Frame-Options: DENY`。
  - 站点页面添加兼容现有 KaTeX / Mermaid / Plotly（`cdn.jsdelivr.net`）及 Turnstile（`challenges.cloudflare.com`）的 `<meta http-equiv="Content-Security-Policy" ...>`，禁止 `object-src`、`base-uri` 劫持与表单外提交（`form-action 'none'`）。

---

## 五、二期评估：浏览器端 WebGPU 自愿启用推理

首期优先确保零新增前端大依赖与全平台（含手机/低配设备）稳定可用；以下作为二期（Phase 2）可选增强的评估结论：

- **候选方案**：WebLLM 或 Transformers.js + Qwen2.5-0.5B/1.5B-Instruct（q4f16）。
- **核心权衡**：
  1. 访客首次开启需下载 **400 MB–1.5 GB** 模型分片，远超站点现有静态资源体积；
  2. 移动端浏览器（尤其 iOS Safari）存在 WebGPU 兼容性差异与显存/内存 OOM 崩溃风险；
  3. WebAssembly / ONNX Runtime 多线程与推理常要求在 CSP `script-src` 中放开 `'wasm-unsafe-eval'`。
- **建议策略**：仅在用户明确点击「离线端侧模型（需下载约 X00 MB）」开关时按需加载，不作为默认或自动回退路径。

---

## 六、站长上线与超额负载验证手册

本 PR 合并前默认以「本地只读段落检索模式」零密钥运行；合入后按以下 4 步启用 Cloudflare Workers AI 云端模式（**全程零费用**）：

### 步骤 1：创建 Turnstile Widget（Cloudflare Dashboard）
1. 进入 Cloudflare Dashboard → **Turnstile** → **Add widget**。
2. Hostnames 填入 `imchong.github.io`（若使用自定义域名请一并添加）。
3. Widget Mode 选择 **Managed**（配合前端 `interaction-only` 无感验证）。
4. 记录 **Site Key**（公开，写入前端配置）与 **Secret Key**（绝不入库）。

### 步骤 2：在本机部署 Worker 并绑定 Secret
```bash
cd workers/kb-chat
npx wrangler login
npx wrangler secret put TURNSTILE_SECRET_KEY   # 粘贴 Turnstile Secret Key
npx wrangler deploy
```
> 部署前确认账户处于 **Workers Free**，且不要开启 Workers Paid 订阅或配置 AI Gateway 付费路由。

### 步骤 3：回填前端公开配置（可选：单独一个小配置 PR）
在站点 HTML 或 `docs/kb-chat.js` 的 `DEFAULT_CONFIG` 中填入公开的 `endpoint`（如 `https://robotics-notebooks-kb-chat.<subdomain>.workers.dev/chat`）与 `turnstileSiteKey`。
- **注意**：若 Worker 使用了自定义域名而非 `*.workers.dev`，需在页面 `<meta http-equiv="Content-Security-Policy">` 的 `connect-src` 中同步加入该自定义源。

### 步骤 4：超额负载与安全负面测试
使用仓库自带脚本对已部署的端点执行验证（支持限速触发与配额耗尽提示检查）：
```bash
# 1. 验证健康检查、越权字段拦截、路径穿越拦截与未带 Turnstile 拒绝
python3 scripts/verify_kb_chat_limits.py --endpoint https://<your-worker>.workers.dev/chat --mode security

# 2. 验证连发触发 429 限速（本地不消费 AI Neurons，因为先被限速或 Turnstile 拦截）
python3 scripts/verify_kb_chat_limits.py --endpoint https://<your-worker>.workers.dev/chat --mode burst --count 15
```
并在 Cloudflare Dashboard → **Workers & Pages** → **Metrics / AI** 确认账户仍为 **Free** 且未触发任何计费；当每日 10,000 Neurons 耗尽时，Worker 返回 `503 QUOTA_OR_AI_UNAVAILABLE`，浏览器自动切换至本地只读段落检索。
