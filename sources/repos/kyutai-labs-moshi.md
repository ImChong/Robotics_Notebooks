# Kyutai Moshi 官方源码 / kyutai-labs/moshi

> 一手官方仓库归档（ingest）

- **仓库：** <https://github.com/kyutai-labs/moshi>
- **论文：** <https://arxiv.org/abs/2410.00037v2>；**Demo / 项目页：** <https://moshi.chat/>
- **模型与权重：** <https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd>
- **核对日期：** 2026-11-15；对照仓库 main 的 README、`moshi/README.md`、`moshi/moshi/server.py`、`moshi/moshi/models/lm.py`、`moshi/moshi/models/loaders.py`。
- **通读：** 已完成
- **开放程度：** 已开源推理栈：PyTorch `moshi/`、MLX `moshi_mlx/`、Rust/Candle `rust/`，也提供 Web 客户端。额外微调入口是 <https://github.com/kyutai-labs/moshi-finetune>。不宣称论文全部原始训练语料/原始完整训练流水线开源。
- **许可：** 官方 README：Python / Web 客户端 MIT；Rust 后端 Apache-2.0；发布的模型权重 CC-BY 4.0（不同后端和权重需分别确认）。

## 官方 PyTorch 交互入口与代码实证

```bash
pip install -U moshi
python -m moshi.server
# 浏览器 localhost:8998；或者 python -m moshi.client
```

官方 `moshi/moshi/server.py`：`ServerState.handle_chat` 接收 WebSocket → `recv_loop` 从客户端 Opus 包解码 PCM 并缓存为 Mimi 帧（24 kHz / 12.5 Hz，1920 samples）→ `mimi.encode(chunk)` → `lm_gen.step(codes[:, :, c:c+1])` → `decode_and_send` 通过 `mimi.decode(tokens[:, 1:])` 还原音频、用 Opus 及文本 token 分别发送到 WebSocket。会话开始时重置 Mimi/LMGen streaming state；`LMGen` 位于 `moshi/moshi/models/lm.py`。

- **运行限制：** 官方 PyTorch README 提示需要约 24 GB GPU 显存（不支持该后端通用量化）；MLX 和 Rust 有不同量化与环境约束；实际延迟受客户端回声消除/链路影响。
- **本次核验方式：** 静态阅读 README 与上述 Python 代码；没有在 GPU、本地麦克风或真实机器人上运行。
- **对应 wiki：** [Moshi 论文节点](../../wiki/entities/paper-moshi-speech-text-realtime-dialogue.md)。
