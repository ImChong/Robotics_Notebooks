---
type: entity
tags: [paper, jepa, vision-language, video-understanding, representation-learning, meta, hkust, nyu]
status: complete
updated: 2026-10-06
arxiv: "2512.10942"
venue: "arXiv 2025"
related:
  - ./article-videodb-jepa-world-models.md
  - ./paper-sa-2603-14482-v-jepa-2-1-unlocking-dense-features-in-video-sel.md
  - ./paper-lejepa.md
sources:
  - ../../sources/papers/vl_jepa_arxiv_2512_10942.md
summary: "VL-JEPA 以视觉表征和文本 query 预测目标文本 embedding，并按需解码为文字；论文报告参数与选择性解码效率收益，但模型不直接输出机器人动作。"
---

# VL-JEPA: Joint Embedding Predictive Architecture for Vision-language

**VL-JEPA**（arXiv:2512.10942）由 Meta FAIR、香港科技大学、Sorbonne Université 与纽约大学研究者提出。它把常见的视觉到文本 token 生成改为预测目标文字的连续 embedding：视觉 encoder 提取画面状态，文本 encoder 编码目标语句，predictor 根据视觉 embedding 和 query 预测目标文本 embedding；需要输出文字时再使用轻量 decoder。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VL-JEPA | Vision-Language Joint-Embedding Predictive Architecture | 在视觉语言 embedding 空间进行目标预测的架构 |
| VLM | Vision-Language Model | 联合处理视觉输入与语言输入/输出的模型 |
| VQA | Visual Question Answering | 视觉问答 |
| JEPA | Joint-Embedding Predictive Architecture | 以 latent target prediction 为核心的架构思路 |

## 方法流程

```mermaid
flowchart TB
  Video["视频 / 图像"] --> Vision["视觉 encoder"]
  Query["文本 query"] --> Predictor["预测目标文本 embedding"]
  Vision --> Predictor
  Target["目标文本"] --> TextEncoder["文本 encoder"]
  TextEncoder --> Predictor
  Predictor --> Decode["按需解码"]
```

训练目标使用目标文本 embedding 作预测目标，而不是逐 token 生成目标句子。嵌入空间也可用于开放词汇分类、视频检索和区分式 VQA 等任务。

## 论文报告的实验

在受控比较中，作者报告与 token-space VLM 相比可训练参数约减少 50%，同时保持或改善指定任务表现；选择性解码在相近输出质量下减少约 2.85 倍解码操作。评估涵盖视频分类、检索和 VQA，具体结果依论文数据集与基线设置。

这项工作不是 action model：论文没有给出可供机器人执行的关节/末端动作策略。它更适合视为视觉语言语义表征路线的一种 JEPA 实例。

## 与其他工作对比

| 工作 | 预测目标 | 与 VL-JEPA 的区别 |
|---|---|---|
| token-space VLM（论文受控基线） | 逐 token 自回归生成目标文本 | VL-JEPA 改为预测目标文本 embedding，按需再解码，报告约 50% 更少可训练参数 |
| [V-JEPA 2.1](./paper-sa-2603-14482-v-jepa-2-1-unlocking-dense-features-in-video-sel.md) | 视频自监督 latent 表征 | 纯视觉自监督；VL-JEPA 以文本 query 与文本 embedding 作为预测条件与目标 |
| [LeJEPA](./paper-lejepa.md) | 视觉 JEPA 表征（SIGReg 约束 embedding） | 关注表征学习配方与理论；VL-JEPA 关注视觉语言任务与解码效率 |
| VLA / 动作条件世界模型 | 机器人动作或动作后果 | VL-JEPA 不输出动作，也不建模动作条件状态转移 |

## 结论

VL-JEPA 表明在视觉语言任务上，把“生成 token”换成“预测文本 embedding”可以在受控比较中减少可训练参数和解码开销，并支持分类、检索与 VQA 共用一个嵌入空间。对机器人而言，它是语义表征层的参考，而不是可直接部署的动作模型或世界模型。

## 关联页面

- [V-JEPA 2.1](./paper-sa-2603-14482-v-jepa-2-1-unlocking-dense-features-in-video-sel.md)
- [LeJEPA](./paper-lejepa.md)
- [VideoDB JEPA 长文](./article-videodb-jepa-world-models.md)

## 参考来源

- [V-JEPA 2.1](./paper-sa-2603-14482-v-jepa-2-1-unlocking-dense-features-in-video-sel.md) — 视频自监督表征工作
- [LeJEPA](./paper-lejepa.md) — 通过 SIGReg 约束 embedding 的 JEPA 配方
- [VideoDB JEPA 长文](./article-videodb-jepa-world-models.md)
- [论文来源归档](../../sources/papers/vl_jepa_arxiv_2512_10942.md)
- [arXiv:2512.10942](https://arxiv.org/abs/2512.10942)
