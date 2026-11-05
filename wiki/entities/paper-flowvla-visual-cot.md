---
type: entity
project_id: flowvla-visual-cot
tags:
  - paper
  - vla
  - world-model
  - optical-flow
  - visual-cot
status: complete
updated: 2026-11-04
arxiv: "2508.18269"
project: https://irpn-lab.github.io/FlowVLA/
related:
  - ../methods/vla.md
  - ../concepts/world-action-models.md
  - ../entities/paper-flowvla-pluggable-3d-flow.md
  - ../entities/paper-flowvla-quantitative-robotic-pouring.md
  - ../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md
sources:
  - ../../sources/papers/flowvla_visual_cot_arxiv_2508_18269.md
  - ../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md
summary: "FlowVLA（Visual CoT）（arXiv:2508.18269）：FlowVLA 提出 Visual Chain of Thought：让 VLA 内部的世界模型先推理运动动力学再生成未来帧，显式链路为 v_t → f_t → v_{t+1}，其中 f_t 是编码运动的中间光流预测。"
---

# FlowVLA（Visual CoT）

**FlowVLA（Visual CoT）**（[arXiv:2508.18269](https://arxiv.org/abs/2508.18269)，[项目页](https://irpn-lab.github.io/FlowVLA/)）是 [新机器视觉《2026年了，主流具身智能算法有哪些？》](https://mp.weixin.qq.com/s/aIYDRlrluZIsmj3fmMsXIg) 点名的算法之一。本页以 arXiv 摘要页为一手资料，文章说法放在「文章说法核对」。

## 一句话定义

**FlowVLA 提出 Visual Chain of Thought：让 VLA 内部的世界模型先推理运动动力学再生成未来帧，显式链路为 v_t → f_t → v_{t+1}，其中 f_t 是编码运动的中间光流预测。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |

## 为什么重要

- 在文章的 2026 算法栈里属于：VLA层；总览见 [2026 具身智能算法栈：VLA、扩散策略、强化学习与世界模型（一手核对版）](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)。

## 一手资料（arXiv 摘要页）

- **标题 / 编号：** [FlowVLA（Visual CoT）](https://arxiv.org/abs/2508.18269)（提交 2025-08-25；（arXiv，v3 更名））
- **作者：** Zhong, Yan, Li, Liu, Gong 等（arXiv 摘要页前 5 位）
- **摘要要点：** 摘要称：许多 VLA 内置的世界模型靠下一帧预测 v_t → v_{t+1}，缺少显式的运动推理，常导致物理上不合理的视觉预测与低效的策略学习；FlowVLA 是自回归 Transformer，显式物化 v_t → f_t → v_{t+1}，使「动力学预测」的预训练目标与「动作生成」的下游任务对齐；在操作基准与真机评测上，既生成更连贯、物理合理的预测，又取得 SOTA 策略性能与显著提升的样本效率。（WebSearch 补：早期版本标题为 FlowVLA: Thinking in Motion with a Visual Chain of Thought；作者来自 HKUST(GZ) 与上海交大，未在摘要页核对机构。）

## 文章说法核对

- **文章：** 文章把 FlowVLA 归为 2026 下半年针对长程与推理的架构创新之一：「引入视觉思维链进行运动推理」。
- **核对：** 摘要与文章一致。库内另有两篇同名 FlowVLA（pluggable-3d-flow、quantitative-robotic-pouring），是别的论文，本节点对应 arXiv:2508.18269。

## 关联页面

- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)
- [paper-flowvla-pluggable-3d-flow](../entities/paper-flowvla-pluggable-3d-flow.md)
- [paper-flowvla-quantitative-robotic-pouring](../entities/paper-flowvla-quantitative-robotic-pouring.md)
- [embodied-algorithm-stack-2026-vla-diffusion-rl-world-model](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)

## 参考来源

- [flowvla_visual_cot_arxiv_2508_18269.md](../../sources/papers/flowvla_visual_cot_arxiv_2508_18269.md)
- [wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md](../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md)
- [arXiv:2508.18269](https://arxiv.org/abs/2508.18269)
