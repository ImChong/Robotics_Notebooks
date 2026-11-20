# Robotics Notebooks 只读 AI 问答 Worker 网关

本目录包含面向 GitHub Pages 静态站的 **零支出、防刷额度、无写入能力** Cloudflare Worker 网关源码（无任何密钥入库）。

## 1. 安全与零支出边界

- **严格停留在 Workers Free 套餐**：账户不升级 Paid、不绑定第三方付费 API Key、不配置 AI Gateway 付费路由。Workers AI Free 每日 `10,000 Neurons` 耗尽后触发硬停止，绝不产生账单。
- **非 Agent 架构**：`env.AI.run()` 不配置任何 `tools` / `functions`，Worker 不持有 GitHub PAT、Deploy Key 或写接口。
- **防匿名刷额度**：先校验请求字节上限（`<= 4 KB`）与严格 JSON Schema，再执行多层限流（边缘 `10s/60s` Rate Limiting + Isolate 单 IP 并发锁与日限额），随后校验 Cloudflare Turnstile `siteverify` 一次性 Token 与站内片段相关性（无关输入消耗 0 Neuron），并缓存公共问答结果。

## 2. 本机部署步骤（不向 GitHub 仓库存放任何 Cloudflare 凭证）

1. 在 Cloudflare Dashboard 创建 **Turnstile Widget**（Managed 模式，Hostname 填 `imchong.github.io`），记录 **Site Key** 与 **Secret Key**。
2. 在本机执行登录、写入 Secret 并部署：
   ```bash
   cd workers/kb-chat
   npx wrangler login
   npx wrangler secret put TURNSTILE_SECRET_KEY
   npx wrangler deploy
   ```
3. 部署后会得到 `https://robotics-notebooks-kb-chat.<your-subdomain>.workers.dev`，将其 `/chat` 地址与 Turnstile Site Key 回填至站点配置（或覆盖 `window.RN_KB_CHAT_CONFIG`）即可启用云端 AI 归纳模式；未配置时前端默认以本地只读段落检索模式运行。
4. 若使用自定义 Worker 域名，请在站点 HTML 的 `<meta http-equiv="Content-Security-Policy">` 的 `connect-src` 中加入对应自定义源。

## 3. 问答记录收集与导出（Cloudflare D1 免费持久化）

Worker 已绑定 D1 数据库 `robotics-notebooks-chat-logs`（只记录时间、语言、问题、回答与命中节点 ID，绝不存储访客 IP、User-Agent 或凭据；D1 异常或超额时静默跳过，不影响问答服务）。

- 初始化远端表结构（首次执行一次）：
  ```bash
  npx wrangler d1 execute robotics-notebooks-chat-logs --remote --file=schema.sql
  ```
- 查看最近 50 条问答：
  ```bash
  npx wrangler d1 execute robotics-notebooks-chat-logs --remote --command "SELECT id, created_at, lang, mode, cached, question, answer, citation_ids FROM chat_logs ORDER BY id DESC LIMIT 50;"
  ```
- 导出全部问答为 JSON（用于定期提炼进 `wiki/queries/`）：
  ```bash
  npx wrangler d1 execute robotics-notebooks-chat-logs --remote --json --command "SELECT * FROM chat_logs ORDER BY id DESC;" > /tmp/chat-logs.json
  ```

## 4. 上线后安全与限流自检

```bash
python3 scripts/verify_kb_chat_limits.py --endpoint https://<your-worker>.workers.dev/chat --mode all
```
