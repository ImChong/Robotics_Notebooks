# MOSS Transcribe Diarize（MOSI 项目页）

- **类型：** 官方产品 / 模型项目页
- **收录日期：** 2026-09-07
- **站点：** <https://mosi.cn/models/moss-transcribe-diarize>
- **在线 Demo：** <https://moss-transcribe-diarize-demo.mosi.cn>
- **API / Pro：** <https://platform.mosi.cn/app/playground>
- **论文：** <https://arxiv.org/abs/2601.01554>
- **代码：** <https://github.com/OpenMOSS/MOSS-Transcribe-Diarize>
- **权重：** <https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize>
- **HF Space：** <https://huggingface.co/spaces/OpenMOSS-Team/MOSS-transcribe-diarize>

## 一句话

**MOSS Transcribe Diarize** 是模思智能（MOSI.AI / OpenMOSS）面向长时多说话人场景的 **SATS（Speaker-Attributed Time-Stamped Transcription）** 旗舰模型族：一次推理输出带 `[Sxx]` 说话人标签与秒级时间戳的结构化转写，可选声学事件标注。

## 为什么值得保留

- 项目页是 **开源边界与 Pro 能力** 的官方入口：0.9B 开源 vs Pro 仅平台 Playground。
- 与 GitHub README、HF 模型卡、技术报告互证 **128k 上下文 / 90 分钟 / 50+ 语言** 等产品表述。

## 站点摘录（2026-09-07 核查）

来源：<https://mosi.cn/models/moss-transcribe-diarize>（同域 mosi.cn 首页描述 MOSS 系列含 Transcribe / TTS / VL）

- **任务：** 会议、播客、访谈、课程、视频等 **长时、 messy、多说话人** 音频 → 结构化转写。
- **输出格式：** `[start_time][Sxx]text[end_time]` 紧凑串联；时间戳单位为秒。
- **模型线：** **MOSS-Transcribe-Diarize 0.9B**（开源 SOTA 定位）与 **Pro**（更高整体性能，开放平台）。
- **里程碑：** 2026-07-09 开源 0.9B；2026-07-14 INTERSPEECH 2026 2nd MLC-SLM 14 语第一；2026-07-22 字幕 Web UI 中英双语。

## 开源核查（2026-09-07）

| 资产 | 状态 |
|------|------|
| GitHub 推理 / 服务 / Web UI | **已开源** |
| HF 权重 OpenMOSS-Team/MOSS-Transcribe-Diarize | **已开源** |
| HF Space 演示 | **已部署**（上传 ≤30 分钟音视频，远程推理） |
| Pro 权重 / API | **部分** — 仅 platform.mosi.cn Playground |
| Podcast / Movies 评测集 | **待发布** — 论文承诺 Hugging Face 开源 |

## 对 wiki 的映射

- 主沉淀：[MOSS Transcribe Diarize](../../wiki/entities/paper-moss-transcribe-diarize.md)
- 论文摘录：[moss_transcribe_diarize_arxiv_2601_01554.md](../papers/moss_transcribe_diarize_arxiv_2601_01554.md)
- 代码：[moss-transcribe-diarize.md](../repos/moss-transcribe-diarize.md)
- Demo Space：[moss-transcribe-diarize-hf-space.md](./moss-transcribe-diarize-hf-space.md)
