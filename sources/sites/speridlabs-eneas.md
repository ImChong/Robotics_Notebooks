# SperidLabs ENEAS 项目页

- **类型**：项目静态站点
- **收录日期**：2026-09-09
- **站点**：<https://speridlabs.com/research/eneas>
- **论文**：<https://arxiv.org/abs/2609.03756>
- **代码：** <https://github.com/speridlabs/eneas>
- **演示：** <https://huggingface.co/spaces/speridlabs/eneas>

## 一句话

**ENEAS**（Embedding-guided Neural Ensemble for Adaptive Segmentation）：文本可提示的 **实例跟踪** 与 **语义发现** 统一方法；相对 SAM 3 强调离屏重识别、近景完整性与雕像/画作/反射等本体判别。

## 开源核查（2026-09-09）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — `speridlabs/eneas`（Apache 2.0） |
| **模型** | HF 自动下载（SeC-4B、grounding）；generic 模式需 Ollama |
| **在线 demo** | HF Space `speridlabs/eneas` |

## 站点摘录要点

- **输入**：文本提示 + 有序视频帧或无序图像集。
- **Instance Tracking**：单实例跟拍；遮挡、离屏回归、极端缩放。
- **Semantic Discovery**：开放概念下发现全部实例并排除 doppelganger。
- **评测**：SA-Co/VEval 子集，SAM 3 官方评测器；实例跟踪 HOTA 26.70 vs 26.51；语义发现 HOTA 9.23 vs 9.19。
- **机构**：SperidLabs（Javier del Pino 等项目负责人）。

## 对 wiki 的映射

- 主沉淀：[ENEAS](../../wiki/entities/paper-eneas.md)
- 原始论文档：[eneas_arxiv_2609_03756.md](../papers/eneas_arxiv_2609_03756.md)
- 代码入口：[eneas.md](../repos/eneas.md)
