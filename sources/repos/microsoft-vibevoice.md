# Microsoft VibeVoice（官方仓库）

> 来源归档

- **标题：** VibeVoice: Open-Source Frontier Voice AI
- **类型：** repo
- **作者：** Microsoft
- **链接：** https://github.com/microsoft/VibeVoice
- **项目页：** https://microsoft.github.io/VibeVoice
- **仓库 HEAD：** [16fb2cb](https://github.com/microsoft/VibeVoice/commit/16fb2cb1217c9934a886e1948ffb06120caa2df5)（2026-10-08）
- **代码协议：** MIT（仓库根目录 LICENSE）
- **入库日期：** 2026-10-09
- **一句话说明：** VibeVoice 是 Microsoft 的开源语音模型系列；截图对应的 VibeVoice-ASR 面向长音频转写，在一次推理中输出说话人、时间戳与文本，并支持用户热词/上下文。
- **关键模型：** [VibeVoice-ASR 7B（Hugging Face）](https://huggingface.co/microsoft/VibeVoice-ASR)，模型卡标注 MIT、51 种语言及 arXiv:2601.18184。
- **为什么值得保留：** 它把 ASR、说话人归因和时间定位整合到长上下文识别中，且提供非流式、流式及 CPU 量化方向；适合会议记录与具身语音接口选型。它不是无算力成本的桌面小工具，官方 GPU 环境要求需仔细核对。

## README / 文档要点（2026-10-08 快照）

- **VibeVoice-ASR：** 官方文档称可在单次处理最多 60 分钟长音频，最长输入受 64K token 限制；输出包含 Who（speaker）、When（timestamp）、What（content），支持自定义 hotwords、超过 50 种语言和 code-switching。
- **运行入口：** 本地 demo：`python demo/vibevoice_asr_gradio_demo.py --model_path microsoft/VibeVoice-ASR --share`；文件推理：`python demo/vibevoice_asr_inference_from_file.py --model_path microsoft/VibeVoice-ASR --audio_files <path>`。文档推荐 NVIDIA Deep Learning Container / CUDA 与 ffmpeg。
- **流式模型：** VibeVoice-ASR-Streaming 在音频到达时逐 chunk 输出；上游公告列出 10 种语言及可定制热词。它与 60 分钟非流式 checkpoint 是不同模型/接口，不要混淆指标与配置。
- **CPU 路线：** README 链接 VibeVoice-ASR-BitNet 与 `microsoft/VibeASR.cpp`，定位为 CPU 量化推理；量化精度、硬件线程数和实时因子需按其独立 README 复测。
- **模型族边界：** 仓库还保留 Realtime-0.5B TTS；但历史 VibeVoice-TTS 代码于 2025-09-05 从该仓库移除。不要把曾发布的 TTS 仓库内容描述为当前主仓内仍完整可用。
- **许可证：** repo 根目录为 MIT；VibeVoice-ASR Hugging Face 模型卡也标注 MIT。流式及其他模型权重仍应逐个检查对应 model card 的许可与使用条款。

## 参考来源

- [上游 README（固定 commit）](https://github.com/microsoft/VibeVoice/blob/16fb2cb1217c9934a886e1948ffb06120caa2df5/README.md)
- [VibeVoice-ASR 官方文档](https://github.com/microsoft/VibeVoice/blob/16fb2cb1217c9934a886e1948ffb06120caa2df5/docs/vibevoice-asr.md)
- [流式 ASR 官方文档](https://github.com/microsoft/VibeVoice/blob/16fb2cb1217c9934a886e1948ffb06120caa2df5/docs/vibevoice-asr-streaming.md)
- [仓库 MIT License](https://github.com/microsoft/VibeVoice/blob/16fb2cb1217c9934a886e1948ffb06120caa2df5/LICENSE)
- [VibeVoice-ASR 模型卡](https://huggingface.co/microsoft/VibeVoice-ASR)
- [ASR 技术报告 arXiv:2601.18184](https://arxiv.org/abs/2601.18184)
- [流式 ASR 技术报告 arXiv:2609.02812](https://arxiv.org/abs/2609.02812)
