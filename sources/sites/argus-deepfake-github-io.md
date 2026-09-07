# ARGUS 深度伪造鉴伪（项目页）

> 来源归档（ingest）

- **标题：** Multi-Agent Forensic Reasoning for Generalizable Deepfake Video Detection
- **短名：** ARGUS（deepfake forensics；非机器人 ARGUS）
- **类型：** site / project-page
- **官方入口：** <https://xavierjiezou.github.io/ARGUS/>
- **代码：** <https://github.com/XavierJiezou/ARGUS>
- **数据集：** <https://huggingface.co/datasets/XavierJiezou/argus-datasets>
- **模型：** <https://huggingface.co/XavierJiezou/argus-models>
- **Demo：** <https://huggingface.co/spaces/XavierJiezou/ARGUS>
- **论文：** <https://arxiv.org/abs/2608.06865>
- **入库日期：** 2026-09-07
- **一句话说明：** BJTU / 清华 / 蚂蚁 ARGUS 官方站：FaceVid-Forensics-100K 概览、四观测 + Judge 管线、OOD 榜单与定性案例。
- **开源状态（2026-09-07 核查）：** **已开源** — 页内链 GitHub + HF Dataset/Models/Space；训练与推理脚本可复现主表。

## 页面公开信息

| 资源 | URL / 状态 |
|------|------------|
| 项目页 | <https://xavierjiezou.github.io/ARGUS/> |
| GitHub | <https://github.com/XavierJiezou/ARGUS> |
| HF Dataset | <https://huggingface.co/datasets/XavierJiezou/argus-datasets> |
| HF Models | <https://huggingface.co/XavierJiezou/argus-models> |
| HF Space | <https://huggingface.co/spaces/XavierJiezou/ARGUS> |
| arXiv | <https://arxiv.org/abs/2608.06865> |

## 数据集摘要（FaceVid-Forensics-100K）

- **规模：** 100K 视频 — 21,075 real + 78,925 fake
- **合成法：** 33 种（face swapping / reenactment / entire-face synthesis，含 Seedance 2.0）
- **协议：** training、in-domain、out-of-domain 划分
- **标注：** 多 MLLM 独立报告 → 维度聚合 → 判决一致解释

## OOD 结果快照（项目页表，7,636 OOD 视频）

| 方法类别 | 代表 | Acc | Recall | F1 |
|----------|------|-----|--------|-----|
| 专用小模型 | TFCU (CVPR'25) | 64.28 | 33.44 | 45.20 |
| 开源 MLLM | Qwen3.6-35B-A3B | 53.17 | 50.05 | 35.72 |
| 闭源 MLLM | Gemini-2.5-Pro | 63.78 | 75.29 | 47.45 |
| Forensics MLLM | VideoVeritas (ICML'26) | 57.87 | 78.96 | 43.22 |
| **ARGUS w/o Video** | 四观测 + Judge（仅报告） | **67.41** | **65.00** | **51.01** |
| **ARGUS w/ Video** | 四观测 + Judge（报告+帧） | **69.87** | **81.82** | **53.28** |

## 对 wiki 的映射

- 论文来源：[`argus_arxiv_2608_06865.md`](../papers/argus_arxiv_2608_06865.md)
- 代码归档：[`argus-deepfake.md`](../repos/argus-deepfake.md)
- 论文实体：[`paper-argus-deepfake-forensics.md`](../../wiki/entities/paper-argus-deepfake-forensics.md)
