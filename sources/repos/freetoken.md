# FreeToken（FlashML-org/FreeToken）

> 来源归档

- **标题：** FreeToken
- **类型：** repo
- **来源：** FlashML
- **链接：** <https://github.com/FlashML-org/FreeToken>
- **论文：** <https://arxiv.org/abs/2608.16157>
- **项目页：** <https://flashml.ai>
- **支持模型：** <https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md>
- **许可：** Apache-2.0
- **入库日期：** 2026-09-15
- **一句话说明：** 边缘原生 MoE serving 引擎：带宽自适应 CPU–GPU 协同、语义锚点 KV 复用、FTW 快速加载；`ft serve` 提供 OpenAI/Anthropic 兼容 API，`ft launch` 接主流 coding agent。
- **沉淀到 wiki：** [`wiki/entities/paper-freetoken.md`](../../wiki/entities/paper-freetoken.md)

---

## 仓库入口（README / docs）

| 组件 | 说明 |
|------|------|
| 安装 | `uv pip install "freetoken[accel]"` 或源码 `uv pip install -e ".[accel]"` |
| 起服务 | `ft serve --model <path-or-hf-id>` → 默认 `127.0.0.1:1919` |
| 终端对话 | `ft shell`（附着已有服务或单进程起引擎） |
| Coding agent | `ft launch claude` / `codex` / `dsh` / `hermes` / `openclaw` / `opencode` |
| 权重转换 | `ft checkpoint` → FTW 快速加载格式（可选，`serve` 可自动检测） |
| 带宽校准 | `ft bench bw` → 为 `hybrid` MoE 策略提供 PCIe/CPU 拆分画像 |
| 文档 | [install](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)、[quickstart](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)、[cli](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)、[models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md) |

## MoE 策略（`--moe-strategy`）

| 策略 | 行为 |
|------|------|
| `fused` | 专家常驻 GPU（需足够 VRAM） |
| `offload` | 专家在主机 RAM，GPU LRU 缓存槽；miss 经 PCIe 拉取 |
| `cpu` | miss 在 CPU 计算 |
| `hybrid` | 每步部分 PCIe 拉取、部分 CPU 算，重叠执行；依赖 `ft bench bw` |
| `auto` | dense → fused；MoE → offload，有带宽画像时 → hybrid |

## 已知良好 checkpoint（节选，完整见 models.md）

| 族 | 代表 HF id |
|----|------------|
| DeepSeek-V4 | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| GLM-5.2 / 5.3-Flash | `nvidia/GLM-5.2-NVFP4`、`RedHatAI/GLM-5.3-Flash-NVFP4` |
| Qwen3.6 MoE | `Qwen/Qwen3.6-35B-A3B`（及 FP8 / NVFP4 变体） |
| Qwen3.8-Flash-Next | `Qwen/Qwen3.8-Flash-Next-FP8` 等 |
| gpt-oss | `openai/gpt-oss-120b`、`openai/gpt-oss-20b` |
| Gemma-4 | `google/gemma-4-26B-A4B-it` 等 |

## 开源边界（截至 2026-09-15）

- **已开源**：CLI、引擎核心、文档与 Apache-2.0 许可；可直接 `serve` HF safetensors。
- **桌面应用**：flashml.ai 分发安装包（与开源 CLI 并列入口）。
- **依赖致谢**：受 mini-sglang、SGLang、vLLM、FlashInfer、flash-linear-attention、LightLLM、llama.cpp 启发或复用代码。
