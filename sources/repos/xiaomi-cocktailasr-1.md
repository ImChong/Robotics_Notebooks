# xiaomi-cocktailasr-1（Xiaomi Research）

- **URL：** <https://github.com/xiaomi-research/xiaomi-cocktailasr-1>
- **类型：** 开源推理示例 / 批量评测脚本 / Demo 音频
- **维护方：** Xiaomi Research（`xiaomi-research`）
- **收录日期：** 2026-09-25
- **Tags：** #asr #ts-asr #speech #llm-asr #cocktail-party #open-source
- **权重：** <https://huggingface.co/Ease3/Xiaomi-CocktailASR-1>（模型定义与权重在 HF，本仓 `trust_remote_code` 加载）
- **论文：** <https://arxiv.org/abs/2609.11274>

## 一句话

**Xiaomi-CocktailASR-1** 官方配套仓：Transformers **`AutoModel.from_pretrained(..., trust_remote_code=True)`** 单条推理、`tools/test_batch_scp.py` 五列 TSV 批量评测、正负样本 **demo/**；**不含** 内联 `modeling_mic_asr.py`（在 HF 模型目录）。

## 为什么值得保留

- 复现 README 基准与 **拒识 / CoT** 行为的 **白盒调用入口**（`model(target, ref)` / `cot=True`）。
- 与人形 **「只听示教者」** 语音链选型直接相关：需 ref  wav 注册操作者声纹。

## 核心内容（结构级）

| 模块 | 说明 |
|------|------|
| HF `Ease3/Xiaomi-CocktailASR-1` | `MicAsrConfig`、`modeling_mic_asr.py`（D2V2 编码器 + LLM）、`feature_extraction_mic_asr.py` |
| Python API | `model("target.wav", "ref.wav")`；内部 ref + 1s 静音 + target 拼接 |
| `tools/test_batch_scp.py` | `--hf_model_dir` + `--input_scp`（utt_id, wav, text, ref_wav, ref_id）；可选 `--cot` |
| `demo/` | 正样本（有转写）与负样本（空输出）示例音频 |

### 依赖（README）

- `torch`, `torchaudio`, `transformers`, `soundfile`
- 推理 dtype 示例：`torch_dtype="bfloat16"` + CUDA

## 相关引用

- [Xiaomi-CocktailASR-1 实体页](../../wiki/entities/paper-xiaomi-cocktailasr-1.md)
- [人形智能语音交互](../../wiki/methods/humanoid-voice-interaction.md)
- [论文摘录](../papers/xiaomi_cocktailasr_1_arxiv_2609_11274.md)
- [HF 权重归档](../sites/xiaomi-cocktailasr-1-hf.md)
