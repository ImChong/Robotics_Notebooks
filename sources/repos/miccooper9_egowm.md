# miccooper9/egowm

> 来源归档（repo）

- **标题：** EgoWM — Egocentric World Models from Internet Priors
- **代码：** <https://github.com/miccooper9/egowm>
- **项目页：** <https://egowm.github.io/>
- **论文：** <https://arxiv.org/abs/2601.15284>
- **权重：** <https://huggingface.co/anuragba/egowm/>
- **类型：** research-code（SVD 等视频扩散骨干上的动作条件推理）
- **首次入库：** 2026-07-26

## 一句话摘要

把预训练视频扩散模型改造成 egocentric 动作条件世界模型的官方代码；当前以 **SVD 导航推理** 为主，训练/其他骨干/操作推理按 README TODO 分批开放。

## 开源边界

| 项 | 状态（入库日） |
|----|----------------|
| SVD 3-DoF / 25-DoF **导航** 推理脚本 | **已发布** |
| HF 微调权重 | **已发布**（`anuragba/egowm`） |
| SCS 指标脚本 | Soon |
| Wan2.1-14B / Cosmos 训练推理 | Soon |
| SVD 训练与 25-DoF **操作** 推理 | Very Soon / Soon |

## 对 wiki 的映射

- [`wiki/entities/paper-egowm-egocentric-world-model.md`](../../wiki/entities/paper-egowm-egocentric-world-model.md)
- [`sources/papers/egowm_arxiv_2601_15284.md`](../papers/egowm_arxiv_2601_15284.md)
- [`sources/sites/egowm-github-io.md`](../sites/egowm-github-io.md)
