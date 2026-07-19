# Kimi K3 API Quickstart（官方文档）

> 原始资料归档（ingest）

- **标题：** Kimi K3 — API Quickstart
- **类型：** course / tutorial（Kimi API Platform 官方接入指南）
- **组织：** 月之暗面（Moonshot AI）
- **原始链接：** <https://platform.kimi.ai/docs/guide/kimi-k3-quickstart>
- **文档索引：** <https://platform.kimi.ai/docs/llms.txt>
- **入库日期：** 2026-07-19
- **一句话说明：** Kimi K3 的 **OpenAI 兼容 API** 接入说明：模型 ID `kimi-k3`、thinking mode、`reasoning_effort`、视觉 / 结构化输出 / 工具调用 / 1M 自动缓存等能力与限制。

## 核心摘录（归纳，非全文）

### 1) 模型摘要

- **2.8T 参数** MoE；**KDA + AttnRes**；原生视觉；**1M context**。
- 首个开源 **3T-class** 模型；权重 **2026-07-27 前** 发布（截至入库日尚未上线公开权重仓库）。
- 场景：**long-horizon coding**、**knowledge work**、**reasoning**。

### 2) 接入方式

- **Base URL：** `https://api.moonshot.ai/v1`
- **鉴权：** 环境变量 `MOONSHOT_API_KEY`
- **SDK：** OpenAI Python SDK（`openai>=1.0`），`base_url` 指向 Moonshot。
- **模型名：** `kimi-k3`

### 3) API 能力要点

| 能力 | 说明 |
|------|------|
| **Thinking** | K3 **始终开启** thinking；用顶层 **`reasoning_effort`**（勿用 K2.x 的 `thinking` 参数）；当前仅支持 **`max`**（默认） |
| **Streaming** | `reasoning_content` 与最终 `content` 分通道 delta |
| **Vision** | `content` 必须为 **对象数组**；图像用 **base64** 或 `ms://<file_id>`；**不支持公网 image URL** |
| **Video** | `client.files.create` 上传后 `video_url: ms://<id>` |
| **Structured output** | `response_format.type=json_schema` + `strict: true`；只解析 `message.content`，非 `reasoning_content` |
| **Partial mode** | assistant 消息设 `partial=True` 从前缀续写 |
| **Tool calling** | `tool_choice="required"` 首回合强制工具；多轮须 **完整 assistant message** 原样回传 |
| **Dynamic tool loading** | 无 `content` 的 `system` 消息内嵌完整 tool 定义，从该位置生效 |
| **Context caching** | **自动**；长前缀不变即可命中，无需 cache ID / TTL 参数 |
| **Official tools** | 经 **Formula** `/tools` + `/fibers` 集成；**web search 近期不推荐生产使用** |

### 4) 重要限制（文档 FAQ / limits）

- `reasoning_effort`：仅 `max`。
- `max_completion_tokens`：默认 **131072**，最大 **1048576**。
- **固定采样参数**（请求中应省略）：`temperature=1.0`、`top_p=0.95`、`n=1`、`presence_penalty=0`、`frequency_penalty=0`。
- 多轮与 tool call：**必须**完整回传 assistant message（含 thinking / tool_calls 等字段）。
- 视觉：`content` 为数组；禁公网图 URL。
- Web search：更新中，近期不建议生产依赖。

### 5) 计费（文档指向 pricing 页）

- **1M context 统一定价**，不按上下文长度分档。
- Input 区分 **cache hit / miss**；output 统一 per-token。

## 对 wiki 的映射

| 目标 | 说明 |
|------|------|
| [Kimi K3](../../wiki/entities/kimi-k3.md) | API 接入、参数约束与工程实践小节 |
| [真机策略 autoresearch 闭环搭建指南](../../wiki/queries/real-robot-policy-autoresearch-harness.md) | 以 Kimi API 驱动 research coding agent 时的接口契约 |

## 外部参考

- [Kimi K3 Quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)
- [Kimi K3 Pricing](https://platform.kimi.ai/docs/guide/kimi-k3-pricing)
- [Kimi K3 技术博客](https://www.kimi.com/blog/kimi-k3)
