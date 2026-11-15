# Moshi：面向实时对话的语音—文本基础模型 / Moshi: a speech-text foundation model for real-time dialogue

> 一手论文归档（ingest）

- **论文：** [arXiv:2410.00037v2](https://arxiv.org/abs/2410.00037v2)；[PDF v2](https://arxiv.org/pdf/2410.00037v2)
- **作者：** Alexandre Défossez、Laurent Mazaré、Manu Orsini、Amélie Royer、Patrick Pérez、Hervé Jégou、Edouard Grave、Neil Zeghidour。
- **机构：** Kyutai。
- **版本：** v2，2024-10-02（v1 2024-09-17）。
- **项目 / Demo：** <https://moshi.chat/>；**官方源码：** <https://github.com/kyutai-labs/moshi>。
- **核对范围：** 论文 v2 的模型/数据/训练（§3–4）、评测与消融（§5，表 3–12）、安全（§6）、附录中的 codec 消融、ASR/TTS 时序、压缩与安全细节；含导论/相关工作和论文结论。论文数据为作者报告，未独立复现实验。
- **通读：** 已完成

## 核心摘录与可追溯结论

1. **并行输入/输出音频流（§3.4.3–3.4.4，图 4）：** 用户和 Moshi 的音频各占一条流，同时建模 Moshi 自己的文本 token；不需要预先切分说话人轮次，允许语音重叠与插话。推理时用户流由真实麦克风音频提供。
2. **Mimi 语义—声学联合编码（§3.3，图 2）：** 24 kHz 波形 → 12.5 Hz 离散帧；语义首码本吸收 WavLM 蒸馏信号，声学码本用于重建；论文模型用 8 个 codebooks，约 1.1 kbps，80 ms 帧。
3. **流式层次预测（§3.4.1–3.4.2，图 3）：** 大 Temporal Transformer 负责跨帧上下文，小 Depth Transformer 逐码本补齐细节；声学码本做一帧延迟，Inner Monologue 令时间对齐文本先于 Moshi 声学 token 预测。
4. **实验与消融（§5.5，表 8）：** 0-shot 语音问答 Web Questions / LlaMA Questions / Audio TriviaQA 的 Moshi 准确率分别为 26.6 / 62.3 / 22.8；关闭 Inner Monologue 后为 9.2 / 21.0 / 7.3。与纯文本 Helium 不同任务/输入形式，不应跨条件直接解读为同水平。
5. **实时性（摘要、§3、§5）：** 理论端到端 160 ms，实际报告约 200 ms（仓库 README 指 L4 GPU 条件），不包含任意机器人外部规划、执行器或网络时延。
6. **派生 ASR/TTS（§5.7，附录 C）：** 通过文本与音频 token 时间先后偏移训练，流式 TTS 和 ASR 在 LibriSpeech test-clean 上分别报告 4.7% 和 5.7% WER；这两个实验使用额外训练设定，不等于直接将对话模型当即插即用的 ASR/TTS 服务。
7. **局限（§5.8、§6）：** 量化后语言能力比主观音质更脆弱；作者评估包含有害内容、训练样本复述、音色控制与水印问题，不表示风险彻底解决。

## 对 wiki 的映射

- [Moshi 独立论文节点](../../wiki/entities/paper-moshi-speech-text-realtime-dialogue.md)
- [人形机器人语音交互](../../wiki/methods/humanoid-voice-interaction.md)

**版权与复现：** 本页只记录事实性摘要和指向原文链接，不转载论文全文；所有指标均为论文作者实验报告。
