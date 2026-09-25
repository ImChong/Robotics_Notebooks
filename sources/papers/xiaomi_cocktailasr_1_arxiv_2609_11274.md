# Xiaomi-CocktailASR-1 Technical Report（arXiv:2609.11274）

> 来源归档（ingest）

- **标题：** Xiaomi-CocktailASR-1 Technical Report
- **类型：** paper / technical report / speech / ASR / TS-ASR / LLM-ASR / cocktail party
- **arXiv：** <https://arxiv.org/abs/2609.11274>（PDF：<https://arxiv.org/pdf/2609.11274.pdf>）
- **机构：** 小米（Xiaomi Research）
- **代码：** <https://github.com/xiaomi-research/xiaomi-cocktailasr-1>
- **权重：** <https://huggingface.co/Ease3/Xiaomi-CocktailASR-1>
- **入库日期：** 2026-09-25
- **一句话说明：** 基于 LLM 的端到端 **目标说话人 ASR（TS-ASR）**：以参考语音作声纹提示，在混合语音中 **无需分离** 直接转写目标说话人；单说话人性能可比主流 ASR，并支持 **负样本拒识** 与可选 **CoT** 推理；LibriMix / AliMeeting / AMI 等多基准 SOTA。

## 开源状态（核查，2026-09-25）

- **已开源：** GitHub 含单条/批量推理示例、`tools/test_batch_scp.py`、`demo/` 正负样本；Apache 2.0。
- **权重与模型代码：** Hugging Face **`Ease3/Xiaomi-CocktailASR-1`**（`trust_remote_code=True` 加载 `modeling_mic_asr.py` 等）；本仓库 **不含** 权重与模型 `.py`（在 HF 包内）。
- **无独立项目页：** 以 arXiv + GitHub README + HF 为官方入口。

## 摘要级要点

- **痛点：** LLM-ASR 多在单说话人场景强，**鸡尾酒会**（多说话人重叠）仍是瓶颈；传统 TS-ASR（说话人 embedding、分离+ASR 级联、早期 LLM 探索）常 **牺牲单说话人精度** 且 **无法在目标不在场时拒识**。
- **方法：** **Xiaomi-CocktailASR-1** — 参考语音 + 1 s 静音 + 混合目标音频拼接输入；**D2V2 音频编码器 + Adapter + LLM** 端到端解码目标说话人文本；内部自动 ref+target 对齐，**不做显式 speech separation**。
- **能力：** ① 多说话人 TS-ASR SOTA；② 单说话人 WER 与 Qwen3-ASR / Whisper 等可比；③ **负样本拒识**（参考说话人不在混合中 → 空输出）；④ **CoT** 模式在 `<answer>` 输出转写、`<think>` 可解释推理。
- **评测（README 表）：** LibriMix 2mix WER **4.11**；AliMeeting Far **20.63**；AMI SDM **21.81**；负样本拒识率 LibriSpeech neg **79.59%**；正样本 **FRR** 在 LibriSpeech 等约 **0.01–0.73%**。

## 核心论文摘录（MVP）

### 1) 参考语音作 voiceprint prompt 的端到端 TS-ASR

- **链接：** Abstract；README「单条推理」
- **摘录要点：** `model(target.wav, ref.wav)` 只转写与 ref 匹配的说话人；无需分离前端，统一架构覆盖单人/多人。
- **对 wiki 的映射：**
  - [Xiaomi-CocktailASR-1](../../wiki/entities/paper-xiaomi-cocktailasr-1.md)
  - [人形智能语音交互](../../wiki/methods/humanoid-voice-interaction.md) — 示教/群聊中「只听操作者」的上游 ASR 选型

### 2) 负样本拒识与单说话人兼容

- **链接：** README「负样本拒识」「单说话人测试集」
- **摘录要点：** 错误 ref 或目标缺席时输出空串，降低误触发；单人集 Non-empty WER 与通用 ASR 同量级，避免 TS 模型只能跑混合轨。
- **对 wiki 的映射：**
  - [Xiaomi-CocktailASR-1](../../wiki/entities/paper-xiaomi-cocktailasr-1.md)
  - [MOSS Transcribe Diarize](../../wiki/entities/paper-moss-transcribe-diarize.md) — 对比：SATS 全说话人转写 vs 本文 **指定目标说话人**

### 3) CoT 与多基准 SOTA

- **链接：** README「思维链（CoT）效果」；Citation
- **摘录要点：** `cot=True` 略降 LibriMix WER；相对 Qwen3-ASR / Gemini / StepAudio 在混合集 WER 差距显著。
- **对 wiki 的映射：**
  - [Xiaomi-CocktailASR-1](../../wiki/entities/paper-xiaomi-cocktailasr-1.md)

## BibTeX

```bibtex
@misc{zhang2026xiaomicocktailasr1technicalreport,
  title={Xiaomi-CocktailASR-1 Technical Report},
  author={Yiru Zhang and Hang Su and Lichun Fan and Ying Zeng and Chang Liu and Yifeng Wang and Yuquan Liang and Tao Li and Lian Li and Wenhao Yang and Jian Luan and Cong Zou and Heng Qu},
  year={2026},
  eprint={2609.11274},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2609.11274}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-xiaomi-cocktailasr-1.md`](../../wiki/entities/paper-xiaomi-cocktailasr-1.md)
- 代码：[`sources/repos/xiaomi-cocktailasr-1.md`](../repos/xiaomi-cocktailasr-1.md)
- 权重：[`sources/sites/xiaomi-cocktailasr-1-hf.md`](../sites/xiaomi-cocktailasr-1-hf.md)
- 互链：[人形智能语音交互](../../wiki/methods/humanoid-voice-interaction.md)、[MOSS Transcribe Diarize](../../wiki/entities/paper-moss-transcribe-diarize.md)
