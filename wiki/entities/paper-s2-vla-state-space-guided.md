---
type: entity
project_id: s2-vla-state-space-guided
tags:
  - paper
  - vla
  - long-horizon
  - manipulation
  - attention
status: complete
updated: 2026-11-04
arxiv: "2606.27872"
related:
  - ../methods/vla.md
  - ../entities/paper-hifor-hierarchical-foresight-vla.md
  - ../entities/paper-s-squared-vla.md
  - ../methods/action-chunking.md
  - ../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md
sources:
  - ../../sources/papers/s2_vla_state_space_guided_arxiv_2606_27872.md
  - ../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md
summary: "S²-VLA（状态空间引导的长程操作）（arXiv:2606.27872）：S²-VLA 针对长程任务中误差累积的问题，引入状态空间引导自适应注意力（SSGAA）：用跟踪任务进度的信念状态生成动态门控权重，自适应融合视觉、任务意图与时序动作三类特征。"
---

# S²-VLA（状态空间引导的长程操作）

**S²-VLA（状态空间引导的长程操作）**（[arXiv:2606.27872](https://arxiv.org/abs/2606.27872)）是 [新机器视觉《2026年了，主流具身智能算法有哪些？》](https://mp.weixin.qq.com/s/aIYDRlrluZIsmj3fmMsXIg) 点名的算法之一。本页以 arXiv 摘要页为一手资料，文章说法放在「文章说法核对」。

## 一句话定义

**S²-VLA 针对长程任务中误差累积的问题，引入状态空间引导自适应注意力（SSGAA）：用跟踪任务进度的信念状态生成动态门控权重，自适应融合视觉、任务意图与时序动作三类特征。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |

## 为什么重要

- 在文章的 2026 算法栈里属于：VLA层；总览见 [2026 具身智能算法栈：VLA、扩散策略、强化学习与世界模型（一手核对版）](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)。

## 一手资料（arXiv 摘要页）

- **标题 / 编号：** [S²-VLA（状态空间引导的长程操作）](https://arxiv.org/abs/2606.27872)（提交 2026-06-26；IJCAI 2026（主会））
- **作者：** Xie, Han, Wei, Sun, Li, Zhao 等（arXiv 摘要页）
- **摘要要点：** 摘要称：VLA 在长程任务中因累积误差传播而性能显著下降，部分原因是静态特征融合用固定权重组合视觉、语言与动作表示，无法适应任务不同阶段；SSGAA 维护信念状态跟踪任务进展，生成动态门控权重，融合视觉特征（空间感知）、任务意图（高层规划）与时序动作序列（执行一致性）；约 2B 参数，持续优于更大的 7B 级模型，在 LIBERO 与 SimplerEnv 等长程操作基准上达到 SOTA。（IJCAI 2026 收录与页码来自 WebSearch，未在 IJCAI 站点核对。）

## 文章说法核对

- **文章：** 文章称「S²-VLA 利用状态空间引导长程操作」。
- **核对：** 摘要与文章一致。**库内已有的 `paper-s-squared-vla`（arXiv:2607.13926）是另一篇驾驶 VLA，不是同一篇**；本节点对应 arXiv:2606.27872。

## 关联页面

- [vla](../methods/vla.md)
- [paper-hifor-hierarchical-foresight-vla](../entities/paper-hifor-hierarchical-foresight-vla.md)
- [paper-s-squared-vla](../entities/paper-s-squared-vla.md)
- [action-chunking](../methods/action-chunking.md)
- [embodied-algorithm-stack-2026-vla-diffusion-rl-world-model](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)

## 参考来源

- [s2_vla_state_space_guided_arxiv_2606_27872.md](../../sources/papers/s2_vla_state_space_guided_arxiv_2606_27872.md)
- [wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md](../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md)
- [arXiv:2606.27872](https://arxiv.org/abs/2606.27872)
