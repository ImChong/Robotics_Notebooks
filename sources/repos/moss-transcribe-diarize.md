# MOSS-Transcribe-Diarize（OpenMOSS）

- **URL：** <https://github.com/OpenMOSS/MOSS-Transcribe-Diarize>
- **类型：** 开源推理 / 服务 / Web UI / 微调
- **维护方：** OpenMOSS / MOSI.AI
- **收录日期：** 2026-09-07
- **Tags：** #asr #speaker-diarization #sats #mllm #speech #open-source
- **权重：** <https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize>
- **论文：** <https://arxiv.org/abs/2601.01554>
- **项目页：** <https://mosi.cn/models/moss-transcribe-diarize>

## 一句话

**0.9B** 端到端 **SATS** 官方实现：Transformers 推理、SGLang Omni / vLLM OpenAI 兼容 API、FunASR 生态集成、字幕 Web App 与 [FINETUNING.md](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/FINETUNING.md) 微调指南。

## 为什么值得保留

- 仓库是复现 **CER / cpCER / Δcp** 榜单与部署长音频的 **唯一白盒入口**；README 含完整架构表与基准数字。
- 多后端（SGLang / vLLM / FunASR）适合作为机器人语音栈 **ASR+diarization 一体化** 选型参考。

## 核心内容（结构级）

| 模块 | 说明 |
|------|------|
| `moss_transcribe_diarize/` | `parse_transcript`、`build_transcription_messages`、`generate_transcription` |
| Python API | `AutoModelForCausalLM` + `AutoProcessor`，Qwen 多模态消息流 |
| SGLang Omni | 推荐 serving；`/v1/audio/transcriptions`，`verbose_json` 解析说话人段 |
| vLLM | 固定 nightly wheel；CUDA 12/13 分索引 |
| FunASR | 生态插件用法（README § Use in the FunASR Ecosystem） |
| Web App | 字幕 UI（中英）；本地或 Docker 部署 |
| FINETUNING.md | 微调流程 |

### 架构规格（README）

| 组件 | 规格 |
|------|------|
| Text backbone | Qwen3-0.6B 风格因果解码器 |
| Audio encoder | Whisper-Medium encoder 配置 |
| Audio frontend | WhisperFeatureExtractor，16 kHz，80 mel，30 s chunk |
| Bridge | 4× temporal merge + MLP adaptor |
| Fusion | 音频特征替换 `<|audio_pad|>` embedding（`masked_scatter`） |

### 环境

- Python **3.12** + Transformers **5.x**（README 实测）
- 可选 `flash-attn`；长音频避免 `eager` attention（内存二次增长）

## 相关引用

- [MOSS Transcribe Diarize 实体页](../../wiki/entities/paper-moss-transcribe-diarize.md)
- [人形智能语音交互](../../wiki/methods/humanoid-voice-interaction.md)
- [项目页归档](../sites/moss-transcribe-diarize-mosi.md)
- [论文摘录](../papers/moss_transcribe_diarize_arxiv_2601_01554.md)
