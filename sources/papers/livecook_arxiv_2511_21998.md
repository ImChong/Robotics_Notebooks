# LiveCook / Qualcomm Interactive Cooking / LiveMamba

- **标题：** Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?
- **类型：** paper
- **arXiv：** [2511.21998](https://arxiv.org/abs/2511.21998)
- **出处：** NeurIPS 2025
- **项目页：** <https://apratimbh.github.io/livecook/>
- **代码：** <https://github.com/Qualcomm-AI-research/qualcomm_interactive_cooking_eval>
- **数据集：** <https://huggingface.co/datasets/qualcomm/qualcomm-interactive-cooking-dataset>
- **入库日期：** 2026-09-15

## 核心摘录

1. **问题设定：** 多模态 LLM 需对视频流**异步**反应，实时给出分步指导、检测步骤完成并**在错误刚出现时**发出纠错反馈——不是 turn-based 对话。
2. **Qualcomm Interactive Cooking：** 基于 CaptainCook4D 构建流式烹饪指导基准与数据集；密集标注**定时** instruction 与 feedback，含精确对齐视觉出现的 mistake alert。
3. **LiveMamba：** 流式多模态 LLM 基线——InternViT 视觉头 → Q-Former 压缩 token → Mamba-130M 语言骨干；可触发 Re-planner 更新后续步骤。
4. **评测双轨：** **Streaming**（多步、误差传播）与 **Turn-based**（单步隔离）并行；指标含 IC-Acc、Mistake Prec/Rec/F1、BERT、ROUGE-L。
5. **数据增强：** ICAug（instruction completion）与 CFAug（counterfactual mistake）显著提升 LiveMamba 的 IC-Acc 与 mistake F1。

## 开源边界（步骤 2.5）

| 已发布 | 未发布 |
|--------|--------|
| 评测代码、HF 数据集、CVPR 2026 入门基线 | LiveMamba 训练/推理代码与权重 |

## 对 wiki 的映射

- 实体页：[wiki/entities/paper-livecook.md](../../wiki/entities/paper-livecook.md)
- 续作：[streaming_interventions_arxiv_2606_09547](streaming_interventions_arxiv_2606_09547.md) → [wiki/entities/paper-streaming-interventions.md](../../wiki/entities/paper-streaming-interventions.md)
