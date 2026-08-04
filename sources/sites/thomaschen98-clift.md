# CLIFT 项目页（thomaschen98.github.io/clift）

> 来源归档

- **标题：** CLIFT: Closed-Loop Iterative Fine-Tuning for Humanoid Specialists
- **类型：** site（项目页）
- **链接：** <https://thomaschen98.github.io/clift>
- **论文：** <https://arxiv.org/abs/2607.29172>（Submitted 2026-07-31）
- **机构：** 加州大学伯克利分校（UC Berkeley）、谷歌 DeepMind（Google DeepMind）、英伟达（NVIDIA Research）
- **入库日期：** 2026-08-04（同日复检）
- **一句话说明：** CLIFT 官方项目页：飞轮示意图（收 rollout → 奖励模型打分 → 优势 token 重标 → 走 SFT API 微调 → 部署）+ 演示视频 + BibTeX；三任务成功率提升直接列在页面上。

## 源码开放核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 代码 | **未发布**——页面列出 Code 条目但**无可用链接**，论文 / 项目页标 `coming_soon`（2026-08-04 复检仍无 GitHub） |
| 权重 | 不适用（GROD 为闭权重，仅通过托管 SFT API 访问） |
| 数据 | 未发布（每任务 2 小时 VR 遥操作演示） |
| 判定 | **宣称将开源 / 截至 2026-08-04 项目页未列 GitHub** |

> **复现门槛提示：** 即便代码放出，核心依赖仍是 **Gemini Robotics On-Device 的托管微调 API 访问权** 与 **Unitree G1 真机 + 全身 VR 遥操作栈**，不是代码可及性问题。

## 页面列出的结果

| 任务 | SFT → 两轮 CLIFT |
|------|------------------|
| Box Packing | 93% → 100% |
| Cup Insertion | 70% → 98% |
| Bimanual Plate Handover | 53% → 96% |

## 相关归档

- [`sources/papers/clift_arxiv_2607_29172.md`](../papers/clift_arxiv_2607_29172.md)
- 沉淀到 wiki：[`wiki/entities/paper-clift-closed-loop-iterative-finetuning.md`](../../wiki/entities/paper-clift-closed-loop-iterative-finetuning.md)
