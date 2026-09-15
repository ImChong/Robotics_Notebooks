# Streaming Interventions / Ego-MC-Bench / Ego-CoMist

- **标题：** Streaming Interventions: Can Video Large Language Models Correct Mistakes as They Occur?
- **类型：** paper
- **arXiv：** [2606.09547](https://arxiv.org/abs/2606.09547)
- **项目页：** <https://apratimbh.github.io/livecookv2/>
- **代码：** <https://github.com/Qualcomm-AI-research/qualcomm_interactive_cooking_eval>
- **Ego-MC-Bench：** <https://huggingface.co/datasets/qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections>
- **Ego-CoMist：** <https://huggingface.co/datasets/qualcomm/qualcomm-interactive-cooking-dataset-counterfactual-mistakes>
- **入库日期：** 2026-09-15

## 核心摘录

1. **Streaming Interventions：** 评估 video LLM 能否在烹饪等日常技能场景中**错误刚出现就主动干预**——既要判「何时说」，也要生成有用纠错话术。
2. **Ego-MC-Bench：** 真厨房场景采集，专家提供 instruction–feedback 对；多视角同步以便 mistakes 一出现即可时间戳标注。
3. **难度：** SOTA video LLM 在 per-recipe step 上 mistake F1 极低（如 Gemini-3-Flash ≈0.18）；任务叠加感知、记忆、时序定位、预期与主动沟通。
4. **Ego-CoMist：** 将**无交互**烹饪视频转为带反事实错误与纠正反馈的监督样本，缓解「有 mistake + 及时干预」训练数据稀缺。
5. **微调收益：** Ego-CoMist+ 显著提升小模型（如 Qwen3.5-2B F1 0.20）的干预能力，利于边缘端低延迟助手。

## 开源边界（步骤 2.5）

| 已发布 | 未发布 |
|--------|--------|
| 评测代码、Ego-MC-Bench、Ego-CoMist HF 数据集 | 反事实合成管线与官方微调脚本 |

## 对 wiki 的映射

- 实体页：[wiki/entities/paper-streaming-interventions.md](../../wiki/entities/paper-streaming-interventions.md)
- 前作：[livecook_arxiv_2511_21998](livecook_arxiv_2511_21998.md) → [wiki/entities/paper-livecook.md](../../wiki/entities/paper-livecook.md)
