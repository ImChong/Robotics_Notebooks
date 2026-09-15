# Streaming Interventions 项目页（apratimbh.github.io/livecookv2）

- **标题：** Streaming Interventions: Can Video Large Language Models Correct Mistakes as They Occur?
- **类型：** site
- **项目页：** <https://apratimbh.github.io/livecookv2/>
- **论文：** [arXiv:2606.09547](https://arxiv.org/abs/2606.09547)
- **代码（评测）：** <https://github.com/Qualcomm-AI-research/qualcomm_interactive_cooking_eval>
- **Ego-MC-Bench 数据集：** <https://huggingface.co/datasets/qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections>
- **Ego-CoMist 数据集：** <https://huggingface.co/datasets/qualcomm/qualcomm-interactive-cooking-dataset-counterfactual-mistakes>
- **博客导读：** <https://apratimbh.github.io/blogs/ego-mc-bench/>
- **入库日期：** 2026-09-15

## 开源核查（入库日）

| 资产 | 状态 |
|------|------|
| 官方评测代码 | **已开源** [qualcomm_interactive_cooking_eval](https://github.com/Qualcomm-AI-research/qualcomm_interactive_cooking_eval)（与 LiveCook 共用） |
| Ego-MC-Bench 真机纠错标注 | **已开源** HF `qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections` |
| Ego-CoMist 反事实合成数据 | **已开源** HF `qualcomm/qualcomm-interactive-cooking-dataset-counterfactual-mistakes` |
| 反事实生成 / 微调训练脚本 | **未发布** — 项目页无独立训练仓；读者可用 HF 数据自行微调 |

## 交叉链接

- 论文摘录：[streaming_interventions_arxiv_2606_09547](../papers/streaming_interventions_arxiv_2606_09547.md)
- 评测仓库：[qualcomm-interactive-cooking-eval](../repos/qualcomm-interactive-cooking-eval.md)
- 前作项目页：[livecook](livecook.md)
- 主实体：[Streaming Interventions](../../wiki/entities/paper-streaming-interventions.md)
