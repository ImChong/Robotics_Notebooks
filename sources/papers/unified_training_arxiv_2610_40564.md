---
type: paper
title: "Training Foundation Models with a Single Algorithm"
status: draft
updated: 2026-10-31
---

# Training Foundation Models with a Single Algorithm

- 论文：[arXiv:2610.40564v1](https://arxiv.org/abs/2610.40564v1)，2026-10-29；[全文](https://arxiv.org/html/2610.40564v1)
- 作者：Yuda Song、Alvin Zhang、Fahim Tajwar、Daman Arora、Jeff Schneider、Daniel Khashabi、Ruslan Salakhutdinov、J. Andrew Bagnell、Aarti Singh、Andrea Zanette
- 机构：卡内基梅隆大学、约翰斯·霍普金斯大学、Aurora Innovation
- [项目主页](https://yudasong.github.io/unified-training/)；[官方代码](https://github.com/yudasong/unified-training)；[模型](https://huggingface.co/sfanm/unified-training)；[数据](https://huggingface.co/datasets/sfanm/unified-training)

## 初步核对

- 将静态语料、示范与适当 RL 目标表述为混合数据分布上的似然拟合；MaxRL 使用成功条件分布作为目标。
- 系统异步生成和训练：生成器评估多个回答，成功样本以估计成功率倒数加权，训练器从缓冲区混合采样并定期发布参数。
- 实验包括 124M 受控推理任务和 0.76B 模型的数学强化学习；配比与算力预算影响准确率、覆盖率和保留能力的取舍，不能概括为所有条件均优于分阶段训练。

## 待完成

本次仅核对论文主要方法和部分实验以及官方 README；尚未通读全部原文与必要附录，也未检查可运行源码调用链。需完成全文与源码阅读、归档项目页及代码、创建含项目流程图和源码时序图的 wiki 节点，并执行 `make ci-preflight`。在此之前 PR 保持草稿。
