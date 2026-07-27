# moonshotai/Kimi-K3（Hugging Face）

> 来源归档（ingest）

- **标题：** Kimi K3 — Hugging Face 模型卡与开放权重
- **类型：** site / model card（Hugging Face Hub）
- **组织：** 月之暗面（Moonshot AI）
- **官方入口：** <https://huggingface.co/moonshotai/Kimi-K3>
- **License：** other / `license_name: kimi-k3` — <https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE>
- **关联仓 / 报告 / 博客：**
  - <https://github.com/MoonshotAI/Kimi-K3>
  - <https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf>
  - <https://www.kimi.com/blog/kimi-k3>
- **入库日期：** 2026-07-27
- **一句话说明：** Kimi K3 **主权重发布面**：`image-text-to-text` 多模态 MoE，**MXFP4** safetensors（**96** 分片，合计约 **1.56 TB**），附 transformers custom code；与 GitHub 仓同日上线。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| **gated** | **否**（`gated: false`） |
| **权重** | **已开源** · `model-00001-of-000096.safetensors` … `000096` |
| **推理辅助代码** | 仓内含 `configuration_kimi_k3.py`、`encoding_k3.py`、`kimi_k3_processor.py`、`kimi_k3_vision_processing.py`、`media_utils.py` 等（`trust_remote_code` / custom_code） |
| **架构类名** | `KimiK3ForConditionalGeneration`（`model_type: kimi_k3`） |
| **训练全栈** | **未随权重仓发布** |
| **镜像** | ModelScope `moonshotai/Kimi-K3`（见 [modelscope-moonshotai-kimi-k3.md](./modelscope-moonshotai-kimi-k3.md)） |

## 模型卡要点（README 归纳）

- **定位：** open-weight、原生多模态 agentic 旗舰；首个开源 **3T-class**。
- **规格：** 总参 **2.8T**，激活 **104B**；上下文 **1M**；**69 KDA + 24 Gated MLA**；Stable LatentMoE **16/896**。
- **量化：** Native **MXFP4** 权重 + **MXFP8** 激活（QAT，自 SFT 起）。
- **部署指针：** API `platform.kimi.ai` 的 `kimi-k3`；自托管推荐 **vLLM** / **SGLang** / **TokenSpeed**。
- **Usage：** thinking **始终开启**；`reasoning_effort` ∈ `{low, high, max}`（默认 `max`）；多轮须回传完整 `reasoning_content` / `tool_calls`。
- **Coding harness：** 推荐 [Kimi Code CLI](https://www.kimi.com/code)。

## 对 wiki 的映射

- [Kimi K3](../../wiki/entities/kimi-k3.md) — 「开源状态 / 工程实践 / 部署」
- [kimi-k3.md（GitHub）](../repos/kimi-k3.md)
- [kimi_k3_tech_report.md](../papers/kimi_k3_tech_report.md)
