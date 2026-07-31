# google-gemini/robotics-samples

- **URL**：<https://github.com/google-gemini/robotics-samples>（默认分支 `main`）
- **类型：** repo / API orchestration samples
- **维护方：** Google Gemini（`google-gemini`）
- **收录日期：** 2026-07-31
- **许可：** Apache-2.0
- **Tags：** #gemini-robotics #embodied-reasoning #live-api #spot #vla-orchestration

## 一句话

Google 发布的 **Gemini Robotics ER** Live API 编排样例仓：用 WebSocket 双向流把 ER agent 接到 **Boston Dynamics Spot / Tinybot / 人类遥操作**，演示 tool calling、语音交互与任务编排——**不是** Gemini Robotics 2 VLA 的训练或权重发布仓。

## 为什么值得保留

- 对应官方「ER 2 代码在 GitHub」声明的可核对入口。
- 工程上可复用 **agent server + embodiment 抽象 + Spot OpenAPI tools** 分层，对照本库「高层 planner / 低层技能接口」选型。
- 与闭源 VLA 权重形成清晰边界：能跑的是 **API 编排**，不是全身策略本地训练。

## 开源核查（2026-07-31）

| 项 | 结论 |
|---|---|
| 仓库可见性 | **已公开**（`main`） |
| 可运行入口 | `Getting Started/gemini_robotics_er.ipynb`；`live-api/agent/server.py`；`live-api/spot/apps/api` |
| VLA / On-Device 权重 | **不在本仓** |
| 依赖 | Gemini API key；Spot 需 Boston Dynamics SDK 与真机/模拟端点 |

## 仓库结构（维护者视角）

| 路径 | 作用 |
|------|------|
| `Getting Started/gemini_robotics_er.ipynb` | ER 入门 notebook |
| `live-api/agent/` | FastAPI Physical Agent Server：Live API、tool dispatch、camera poller |
| `live-api/agent/embodiment/spot/` | Spot embodiment + OpenAPI tools |
| `live-api/spot/` | Spot SDK 应用：navigation / manipulation / hydration REST |
| `live-api/tinybot/` | 小型硬件相机流与 REST 控制 |

## Quickstart（摘自上游 README）

```bash
cd live-api/agent
UV_CACHE_DIR=.uv-cache uv sync
export GEMINI_API_KEY="your_api_key_here"
UV_CACHE_DIR=.uv-cache uv run python server.py --port 8000
```

## 对 wiki 的映射

- [gemini-robotics](../../wiki/entities/gemini-robotics.md) — 「源码运行时序图」对齐本仓路径
- [gemini_robotics_2_whole_body](../blogs/gemini_robotics_2_whole_body.md)
- [gemini-robotics 产品页](../sites/gemini-robotics.md)
