# Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities（arXiv:2505.17862）

> 来源归档（ingest）

- **标题：** Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities
- **类型：** paper / benchmark / audio-visual QA / omni-modal MLLM / temporal alignment
- **arXiv：** <https://arxiv.org/abs/2505.17862>（PDF：<https://arxiv.org/pdf/2505.17862.pdf>）
- **作者：** Ziwei Zhou、Rui Wang、Zuxuan Wu*、Yu-Gang Jiang*
- **机构：** 复旦大学（计算与人工智能创新学院；可信具身智能研究院）
- **项目页：** <https://lliar-liar.github.io/Daily-Omni/>（含 [#leaderboard](https://lliar-liar.github.io/Daily-Omni/#leaderboard)）
- **代码：** <https://github.com/Lliar-liar/Daily-Omni>（**GPL-3.0**）
- **数据集：** <https://huggingface.co/datasets/liarliar/Daily-Omni>（**CC BY-NC-SA 4.0**；含 `Videos.tar` ≈ 3.9 GB + `qa.json`）
- **入库日期：** 2026-07-30
- **一句话说明：** 684 段日常视频、1197 道四选一 AVQA，覆盖 6 类跨模态时序推理任务；半自动标注管线 + 24 模型 / 37 模态设定评测；诊断基线 Daily-Omni Agent；榜首含智元 X-Lab WITA-Omni Preview。

## 开源状态（核查，2026-07-30）

- **已开源：** 项目页徽章链到 GitHub + Hugging Face；仓库含 QA 生成管线（`run_pipeline.py`）、评测脚本（`test_model/` / `test_model_api/`）、诊断基线（`baseline/`）与示例视频。
- **数据：** HF `liarliar/Daily-Omni` 公开 `Videos.tar` + `qa.json`（非门控）；许可证 **CC BY-NC-SA 4.0**（非商业复用需注意）。
- **代码许可证：** GitHub API `license.spdx_id = GPL-3.0`。
- **边界：** 复现评测需自备各模型权重 / API Key；榜单持续更新（含闭源 Preview 模型），分数以项目页 leaderboard 为准。

## 摘要级要点

- **问题：** 现有 MLLM 在视觉/音频单模态基准上表现强，但 **跨模态同步时序推理** 仍缺高质量、可扩展评测；既有 AVQA 常偏领域（音乐/全景）、静态图–音对或窄任务。
- **基准：** 684 真实视频（11 YouTube 类目；AudioSet / Video-MME / FineVideo 采样）→ 30s/60s 片段；**1197** 道 MCQA；六任务族：AV Align / Event Sequence / Reasoning / Inference / Comparative / Context Understanding；随机猜中率 25%。
- **管线：** 分段独立音视频标注 → 一致性修订 → 跨模态事件对齐 → Deepseek-R1 出题 → 文本泄漏过滤（≈47% 丢弃）→ 人工验收（≈30% 接受率，单人 <30h）。
- **评测：** 24 个基础模型、37 个模型–模态设定（AV / V-only / A-only / text-only）；训练无关模块化基线 **Daily-Omni Agent**（Qwen2.5-VL + Qwen2-Audio + Whisper + Qwen2.5-14B）Avg **61.82%**。
- **榜单读法（项目页 2026-07-26 更新）：** AV 全模态 Avg 榜首 **AGIBOT X-Lab WITA-Omni Preview（Closed）85.21%**；开源权重榜首约 **NVIDIA Nemotron 3 Nano Omni 30B A3B 74.52%**；去掉任一模态常掉 **10–28** 个百分点。

## 核心论文摘录（MVP）

### 1) 跨模态时序对齐是主瓶颈

- **链接：** Abstract；§4.2；§5
- **摘录要点：** 许多端到端 OLM 在 alignment-critical 题上挣扎；显式对齐的解耦基线可超过若干近期开源 omni 模型，说明统一架构中时序对齐机制仍不足。
- **对 wiki 的映射：**
  - [Daily-Omni](../../wiki/entities/paper-daily-omni.md)
  - [具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — ① 层「大脑认知」的 **音视频同步** 补维

### 2) 可扩展半自动 QA 管线

- **链接：** §3.2；Figure 3–4
- **摘录要点：** Gemini 分段标注 + 修订 + 事件对齐（人工抽检 100 视频对齐正确率 >90%）+ Reasoning LLM 出题/优化 + 双 LLM 文本泄漏过滤 + 人工验收；支撑后续扩库。
- **对 wiki 的映射：**
  - [Daily-Omni](../../wiki/entities/paper-daily-omni.md)
  - [RoboBench](../../wiki/entities/robo-bench.md) — 同属 MLLM QA 评测，但 RoboBench 面向操纵认知链

### 3) 模态消融证明双通道必需

- **链接：** §4.4.1
- **摘录要点：** Gemini 2.5 Flash 全模态 73.06% → audio-only 54.05% / visual-only 44.61%；Qwen3-Omni 去任一模态掉约 13–16%；音频-only 常高于视觉-only，说明基准非「视觉偏置视频 QA」。
- **对 wiki 的映射：**
  - [Daily-Omni](../../wiki/entities/paper-daily-omni.md)
  - [具身大模型分类学选型闭环](../../wiki/queries/embodied-fm-taxonomy-loop.md) — ① VLM/OLM 感知层 I/O 边界

## BibTeX

```bibtex
@misc{zhou2026dailyomniaudiovisualreasoningtemporal,
  title={Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities},
  author={Ziwei Zhou and Rui Wang and Zuxuan Wu and Yu-Gang Jiang},
  year={2026},
  eprint={2505.17862},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2505.17862}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-daily-omni.md`](../../wiki/entities/paper-daily-omni.md)
- 项目页归档：[`sources/sites/daily-omni-github-io.md`](../sites/daily-omni-github-io.md)
- 代码归档：[`sources/repos/daily-omni.md`](../repos/daily-omni.md)
- 互链：[RoboBench](../../wiki/entities/robo-bench.md)、[ESI-Bench](../../wiki/entities/esi-bench.md)、[具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)、[具身大模型分类学选型闭环](../../wiki/queries/embodied-fm-taxonomy-loop.md)
