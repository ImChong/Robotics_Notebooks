# MOSS Transcribe Diarize Technical Report（arXiv:2601.01554）

> 来源归档（ingest）

- **标题：** MOSS Transcribe Diarize Technical Report
- **类型：** paper / technical report / speech / ASR / speaker diarization / MLLM / SATS
- **arXiv：** <https://arxiv.org/abs/2601.01554>（PDF：<https://arxiv.org/pdf/2601.01554.pdf>）
- **机构：** MOSI.AI（模思智能 / OpenMOSS 团队；顾问 Xipeng Qiu，复旦大学 NLP 组）
- **项目页：** <https://mosi.cn/models/moss-transcribe-diarize>
- **在线 Demo：** <https://moss-transcribe-diarize-demo.mosi.cn> · HF Space <https://huggingface.co/spaces/OpenMOSS-Team/MOSS-transcribe-diarize>
- **代码：** <https://github.com/OpenMOSS/MOSS-Transcribe-Diarize>
- **权重：** <https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize>
- **入库日期：** 2026-09-07
- **一句话说明：** 0.9B 统一 MLLM，端到端完成 Speaker-Attributed Time-Stamped Transcription（SATS）；128k 上下文可单次处理最长约 90 分钟多说话人音频；在 AISHELL-4 / Podcast / Movies 上优于多家闭源商用系统。

## 开源状态（核查，2026-09-07）

- **已开源：** GitHub 仓库含推理、Web UI、SGLang/vLLM 服务、FunASR 集成与微调文档；HF 公开 **MOSS-Transcribe-Diarize 0.9B** 权重（`trust_remote_code=True`）。
- **部分闭源：** **MOSS Transcribe Diarize Pro** 更强版本仅通过 [MOSI 开放平台 Playground](https://platform.mosi.cn/app/playground) 提供，不在 GitHub/HF 权重库。
- **评测数据：** 论文称 Podcast / Movies 内部集将开源至 Hugging Face（入库日 README 已链 HF 模型与 Space，数据集页以 upstream 为准）。
- **边界：** 长音频需调高 `max_new_tokens`；`eager` attention 对长序列 OOM；SGLang Omni 当前偏 CUDA 13 环境。

## 摘要级要点

- **任务 SATS：** 同时输出「说了什么、谁说的、何时说的」——会议转写、呼叫中心、法律取证等场景的核心需求。
- **痛点：** 传统 ASR（Whisper）+ 说话人分离（Pyannote/x-vector）级联管线错误级联、短上下文分块导致身份漂移、难原生输出段级时间戳。
- **方法：** 音频编码器（Whisper-Medium 配置）+ 4× 时序 merge + MLP 投影至 **Qwen3-0.6B** 文本骨干；时间戳以格式化文本 token 插入，支持 **128k** 上下文（最长约 **90 分钟**）单次前向。
- **训练数据：** 互联网多语真实对话（含 AISHELL-4 远场）+ 可控概率模拟混合（2–12 说话人、重叠、噪声混响 SNR 0–15 dB）。
- **指标：** CER（纯 ASR）、cpCER（带说话人置换的最小编辑距离）、Δcp = cpCER − CER（隔离分离错误）。
- **结果（技术报告 Table 2）：** 在 AISHELL-4 / Podcast / Movies 上 CER、cpCER、Δcp 均优于 Doubao、ElevenLabs、Gemini 2.5 Pro 等；GPT-4o / Gemini 3 Pro 对长音频格式不稳定或未完整评测。
- **工程：** 2026-07-09 开源 0.9B；2026-07-14 获 INTERSPEECH 2026 **2nd MLC-SLM Challenge** 14 语第一名；支持 50+ 语言。

## 核心论文摘录（MVP）

### 1) 端到端 SATS 统一建模

- **链接：** §1–2；Figure 2
- **摘录要点：** 单次前向联合词识别、说话人归因与时间戳预测；避免 DiarizationLM / Sortformer 两阶段或 Whisper+Pyannote 级联的跨模块失配。
- **对 wiki 的映射：**
  - [MOSS Transcribe Diarize](../../wiki/entities/paper-moss-transcribe-diarize.md)
  - [人形智能语音交互](../../wiki/methods/humanoid-voice-interaction.md) — ASR 上游可替换为 SATS 一体输出

### 2) 128k 长上下文与会议尺度

- **链接：** §1 贡献；§2 时间戳文本编码
- **摘录要点：** 避免 JEDIS-LLM 等分块+Speaker Prompt Cache 的边界伪影；维持长程指代与说话人一致性。
- **对 wiki 的映射：**
  - [MOSS Transcribe Diarize](../../wiki/entities/paper-moss-transcribe-diarize.md)
  - [Daily-Omni](../../wiki/entities/paper-daily-omni.md) — 同属音频 MLLM 栈，但 Daily-Omni 评 AV 对齐而非转写分离

### 3) 真实+模拟混合训练与 Δcp 评测

- **链接：** §3–4；Table 1–2
- **摘录要点：** 模拟器控制重叠/交替/声学；Δcp 低说明分离不拖累转写；Movies 短句高重叠场景仍保持较小 CER–cpCER 间隙。
- **对 wiki 的映射：**
  - [MOSS Transcribe Diarize](../../wiki/entities/paper-moss-transcribe-diarize.md)

## BibTeX

```bibtex
@misc{yu2026mosstranscribediarizetechnicalreport,
  title={MOSS Transcribe Diarize Technical Report},
  author={Donghua Yu and Zhengyuan Lin and Chen Yang and others},
  year={2026},
  eprint={2601.01554},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.01554}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-moss-transcribe-diarize.md`](../../wiki/entities/paper-moss-transcribe-diarize.md)
- 项目页：[`sources/sites/moss-transcribe-diarize-mosi.md`](../sites/moss-transcribe-diarize-mosi.md)
- 代码：[`sources/repos/moss-transcribe-diarize.md`](../repos/moss-transcribe-diarize.md)
- HF Demo：[`sources/sites/moss-transcribe-diarize-hf-space.md`](../sites/moss-transcribe-diarize-hf-space.md)
- 互链：[人形智能语音交互](../../wiki/methods/humanoid-voice-interaction.md)、[Daily-Omni](../../wiki/entities/paper-daily-omni.md)、[OpenMOSS Awesome-WAM](../../wiki/concepts/world-action-models.md)
