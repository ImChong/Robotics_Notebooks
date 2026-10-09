# VibeVoice-ASR-Streaming Technical Report（arXiv:2609.02812）

> 来源归档（ingest）

- **标题：** VibeVoice-ASR-Streaming Technical Report
- **类型：** paper / technical report / streaming ASR / diarization
- **作者：** Yujie Tu, Zhiliang Peng, Jianwei Yu, et al.（Microsoft）
- **arXiv：** https://arxiv.org/abs/2609.02812（v2，2026-09-10）
- **项目页：** https://microsoft.github.io/VibeVoice
- **代码：** https://github.com/microsoft/VibeVoice
- **权重：** README 链接 VibeVoice-ASR-Streaming 1.5B / 7B；各模型卡许可请分别核对
- **入库日期：** 2026-10-09
- **一句话说明：** 将定长音频块、少量 lookahead 音频和历史文本交织输入，实现边听边输出说话人归因转写，面向实时语音助手与 agent 的低延迟需求。

## 核心论文摘录（MVP）

### 1) 流式 speaker-attributed ASR

- **摘录要点：** 与离线 VibeVoice-ASR 区别在于输入还在到达时便逐块转写，不再等待整段录音完成；在同一模型中生成 “who said what”，避免独立 diarization 阶段。
- **对 wiki 的映射：**
  - [Microsoft VibeVoice](../../wiki/entities/microsoft-vibevoice.md)
  - [VibeVoice-ASR 技术报告](vibevoice_asr_arxiv_2601_18184.md)

### 2) 分块与 lookahead 设计

- **摘录要点：** 模型交织固定大小 audio chunks、少量 lookahead 音频与先前文本，在延迟和识别上下文之间取舍。论文摘要报告 7B 模型在五个评测集上的 WER/CER 平均表现及 13 项归因设置的结果；复现比较应以完整协议为准。
- **对 wiki 的映射：**
  - [Microsoft VibeVoice](../../wiki/entities/microsoft-vibevoice.md) — 与离线版本分别评估延迟、字错率与 speaker attribution。

## BibTeX

```bibtex
@misc{tu2026vibevoiceasrstreamingtechnicalreport,
  title={VibeVoice-ASR-Streaming Technical Report},
  author={Tu, Yujie and Peng, Zhiliang and Yu, Jianwei and others},
  year={2026},
  eprint={2609.02812},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2609.02812}
}
```
