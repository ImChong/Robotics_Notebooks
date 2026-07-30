# Daily-Omni（项目页）

> 来源归档（ingest）

- **标题：** Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities
- **类型：** site / project-page / leaderboard
- **官方入口：** <https://lliar-liar.github.io/Daily-Omni/>
- **Leaderboard：** <https://lliar-liar.github.io/Daily-Omni/#leaderboard>
- **代码：** <https://github.com/Lliar-liar/Daily-Omni>
- **数据集：** <https://huggingface.co/datasets/liarliar/Daily-Omni>
- **论文：** <https://arxiv.org/abs/2505.17862>
- **入库日期：** 2026-07-30
- **一句话说明：** 复旦 Daily-Omni 官方站点：QA 示例、生成管线、诊断 Agent、失败案例与持续更新的 omni-modal leaderboard。
- **开源状态（2026-07-30 核查）：** **已开源** — 页头徽章同时列出 arXiv / GitHub / Dataset；代码 GPL-3.0，数据 CC BY-NC-SA 4.0。

## 页面公开信息

| 资源 | URL / 状态 |
|------|------------|
| 项目页 | <https://lliar-liar.github.io/Daily-Omni/> |
| Leaderboard | <https://lliar-liar.github.io/Daily-Omni/#leaderboard> |
| GitHub | <https://github.com/Lliar-liar/Daily-Omni> |
| Hugging Face Dataset | <https://huggingface.co/datasets/liarliar/Daily-Omni> |
| arXiv | <https://arxiv.org/abs/2505.17862> |

## Leaderboard 摘要（入库日快照，AV 全模态 Avg）

随机猜中率 25%。缩写：`AV Align` / `Comp.` / `Ctx. Und.` / `Evt. Seq.` / `Infer.` / `Reas.`；另报 30s / 60s 子集。

| 排名读法 | 模型 | Avg |
|----------|------|-----|
| 闭源榜首 | AGIBOT X-Lab WITA-Omni Preview (Closed) | **85.21** |
| 次席闭源 | Qwen3.5-Omni-Plus (Closed)† | 84.68 |
| 开源权重榜首 | NVIDIA Nemotron 3 Nano Omni 30B A3B (Open) | 74.52 |
| 诊断基线 | Daily-Omni-Baseline-Qwen2.5 (Open) | 61.82 |

† Qwen3.5-Omni-Plus：1197 题中 1175 有效回答（22 题因请求体积/内容过滤无法评）；分母为各子集有效回复。

News（README / 项目页口径，2026-07-26）：新增 WITA-Omni Preview、Qwen3.5-Omni-Plus、Gemini 3.1 Pro Preview、Doubao Seed 2.0 Lite；感谢 AGIBOT 提供部分评测结果。

## 对 wiki 的映射

- 仓库归档：[`daily-omni.md`](../repos/daily-omni.md)
- 论文来源：[`daily_omni_arxiv_2505_17862.md`](../papers/daily_omni_arxiv_2505_17862.md)
- 论文实体：[`paper-daily-omni.md`](../../wiki/entities/paper-daily-omni.md)
