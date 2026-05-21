# mimic-video（mimic-video.github.io）

> 来源归档（ingest）

- **标题：** mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs（项目主页）
- **类型：** project site
- **官方入口：** <https://mimic-video.github.io/>
- **论文：** <https://arxiv.org/abs/2512.15692>
- **机构索引（页面脚注）：** [mimic robotics](https://www.mimicrobotics.com/)、Microsoft Zurich、ETH Zurich、ETH AI Center、UC Berkeley
- **入库日期：** 2026-05-17
- **一句话说明：** 官方项目页：VAM 叙事摘要、**Cosmos-Predict2** 视频骨干 + **部分去噪 \(\tau_v\)** 潜计划 + 本体条件动作解码器的文字版方法说明、**Franka + mimic 灵巧手** 双臂真机对比 **Diffusion Policy** 基线片段、**SIMPLER / LIBERO** 仿真效率曲线叙述，以及 **BibTeX** 与 **NVIDIA Cosmos** 生态外链。

## 页面结构（检索自 2026-05-17 公开站点）

| 区块 | 内容要点 |
|------|----------|
| Abstract | 与 arXiv 摘要一致的「静态 VLM 先验 vs 视频动力学先验」论点与 **10× / 2×** 效率宣称 |
| The Model | **Nvidia Cosmos-Predict2**、独立 \(\tau_v\) / \(\tau_a\) 流日程、潜视觉计划条件化小解码器 |
| Real World | 双臂 **package sorting**、**tape stowing**；每 chunk **潜视频计划 → 执行**；附 DP 基线对比视频区 |
| Sim benchmarks | **SIMPLER**、**LIBERO**；对比「传统 VLA + FAST 预训练 + Knowledge-Insulation 动作头」类基线的叙述 |
| BibTeX | 与论文条目一致（arXiv:2512.15692） |

## 对 wiki 的映射

- [mimic-video（Video-Action Model）](../../wiki/methods/mimic-video.md) — 读者向归纳与与其他路线的关系
- 技术细节以 [sources/papers/mimic_video_arxiv_2512_15692.md](../papers/mimic_video_arxiv_2512_15692.md) 为准
