# Audio8 TTS 官方仓库

> 来源归档（ingest）

- **仓库：** <https://github.com/Edge0-AI/Audio8_TTS>
- **版本：** master 快照 4eb54a69113eedc65677cbc09a30aee809bb39ee，核查日期 2026-11-01。
- **许可：** 仓库及 0.6B 模型 Apache-2.0；0.1B 模型另有许可。
- **开放状态：** 推理、数据准备、监督微调及部署代码公开；Realistic 0.6B 权重见 [模型卡](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b)。不据此断言论文完整训练数据和 GRPO 训练实现均已公开。
- **入口：** `audio8_tts_infer.py`（单句或 JSONL 批量）、`audio8_tts_prepare.py`（训练音频编码）、`audio8_tts_sft.py` / `audio8_tts_sft.sh`（监督微调）。运行 Realistic 须显式设置模型 ID，脚本默认指向 Preview。
- **源码核查：** 推理入口将文本与可选参考音频和准确转写交给 AutoProcessor，调用模型 generate 与 decode_audio，最后经 soundfile 写出。批量失败时回退到逐条推理。
- **关联实体：** [Audio8 TTS](../../wiki/entities/audio8-tts.md)
