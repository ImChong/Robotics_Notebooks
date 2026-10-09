# VIBEVOICE-ASR Technical Report（arXiv:2601.18184）

> 来源归档（ingest）

- **标题：** VIBEVOICE-ASR Technical Report
- **类型：** paper / technical report / ASR / diarization / timestamping
- **作者：** Zhiliang Peng, Jianwei Yu, Yaoyao Chang, et al.（Microsoft）
- **arXiv：** https://arxiv.org/abs/2601.18184（v2，2026-03-14）
- **项目页：** https://microsoft.github.io/VibeVoice
- **代码：** https://github.com/microsoft/VibeVoice
- **权重：** https://huggingface.co/microsoft/VibeVoice-ASR（模型卡标注 MIT）
- **入库日期：** 2026-10-09
- **一句话说明：** VibeVoice-ASR 针对会议、播客等长时多说话人音频，以单次最多 60 分钟处理联合输出文字、说话人和时间戳，并支持 50+ 语言与提示式上下文注入。

## 开源状态与版本核查

截至 2026-10-09，官方 GitHub 主仓包含 ASR 文档、推理入口和训练/微调资源，仓库 License 为 MIT；ASR 权重由 Microsoft 在 Hugging Face 发布，模型卡亦标 MIT。代码快照与局限见 [仓库归档](../repos/microsoft-vibevoice.md)。论文摘要为方法声明，长音频性能仍应在目标数据域复测。

## 核心论文摘录（MVP）

### 1) 统一长音频说话人归因转写

- **摘录要点：** 单个端到端生成模型统一 ASR、speaker diarization 与 timestamping；报告将长音频单次处理上限设为 60 分钟，针对短块切分带来的上下文碎片与多说话人复杂度。
- **对 wiki 的映射：**
  - [Microsoft VibeVoice](../../wiki/entities/microsoft-vibevoice.md)
  - [人形智能语音交互](../../wiki/methods/humanoid-voice-interaction.md)

### 2) 多语言与上下文提示

- **摘录要点：** 报告称支持 50+ 语言、不要求显式语言设置并可处理语码切换；通过 prompt-based context injection 接受定制上下文，以改善领域术语和同音词辨识。
- **对 wiki 的映射：**
  - [Microsoft VibeVoice](../../wiki/entities/microsoft-vibevoice.md) — 目标语言/领域与热词应作为部署评测维度

## BibTeX

```bibtex
@misc{peng2026vibevoiceasrtechnicalreport,
  title={VIBEVOICE-ASR Technical Report},
  author={Peng, Zhiliang and Yu, Jianwei and Chang, Yaoyao and others},
  year={2026},
  eprint={2601.18184},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2601.18184}
}
```
