---
type: entity
project_id: two-steps-diffusion-policy
tags:
  - paper
  - diffusion-policy
  - imitation-learning
  - inference-efficiency
status: complete
updated: 2026-11-04
arxiv: "2510.21991"
related:
  - ../methods/diffusion-policy.md
  - ../entities/paper-diffusion-policy.md
  - ../methods/imitation-learning.md
  - ../methods/action-chunking.md
  - ../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md
sources:
  - ../../sources/papers/two_steps_diffusion_policy_arxiv_2510_21991.md
  - ../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md
summary: "Two-Steps Diffusion Policy（arXiv:2510.21991）：Two-Steps Diffusion Policy 针对具身任务动作分布低维、结构化的特点定制去噪过程，使扩散策略仅需少量神经函数评估（NFE）即可有效运行，并提出 population-based 的 genetic denoising 提升性能与稳定性。"
---

# Two-Steps Diffusion Policy

**Two-Steps Diffusion Policy**（[arXiv:2510.21991](https://arxiv.org/abs/2510.21991)）是 [新机器视觉《2026年了，主流具身智能算法有哪些？》](https://mp.weixin.qq.com/s/aIYDRlrluZIsmj3fmMsXIg) 点名的算法之一。本页以 arXiv 摘要页为一手资料，文章说法放在「文章说法核对」。

## 一句话定义

**Two-Steps Diffusion Policy 针对具身任务动作分布低维、结构化的特点定制去噪过程，使扩散策略仅需少量神经函数评估（NFE）即可有效运行，并提出 population-based 的 genetic denoising 提升性能与稳定性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |

## 为什么重要

- 在文章的 2026 算法栈里属于：扩散策略 / 模仿学习层；总览见 [2026 具身智能算法栈：VLA、扩散策略、强化学习与世界模型（一手核对版）](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)。

## 一手资料（arXiv 摘要页）

- **标题 / 编号：** [Two-Steps Diffusion Policy](https://arxiv.org/abs/2510.21991)（提交 2025-10-24；NeurIPS 2025）
- **作者：** Clemente, Brunswic, Yang, Zhao, Khalil 等（arXiv 摘要页前 5 位）
- **摘要要点：** 摘要称：扩散策略在模仿专家示范上达到 SOTA，但许多推理策略直接从视觉任务迁移而来；针对动作分布低维、结构化的特点定制去噪，可让扩散策略在少至 5 次 NFE 下有效运行；在此基础上提出 genetic denoising，选择分布外风险低的去噪轨迹；方法仅用 2 次 NFE 就解决困难任务且性能提升或持平；在 D4RT / Robomimic 的 14 项操作任务（摘要写 D4RL and Robomimic）上、超过 200 万次评测中持续优于标准扩散策略，最高约 20% 性能提升。

## 文章说法核对

- **文章：** 文章把 Two-Steps Diffusion Policy 与「各类实时加速器」并列为 2026 年效率优化重点：仅需 5 次函数评估，目标是保留鲁棒性的同时压低部署延迟。
- **核对：** 文章写「仅需 5 次函数评估」是摘要的第一个表述；摘要同时称 2 次 NFE 即可解决困难任务，论文方法是 genetic denoising 而非 RL 微调。

## 关联页面

- [diffusion-policy](../methods/diffusion-policy.md)
- [paper-diffusion-policy](../entities/paper-diffusion-policy.md)
- [imitation-learning](../methods/imitation-learning.md)
- [action-chunking](../methods/action-chunking.md)
- [embodied-algorithm-stack-2026-vla-diffusion-rl-world-model](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)

## 参考来源

- [two_steps_diffusion_policy_arxiv_2510_21991.md](../../sources/papers/two_steps_diffusion_policy_arxiv_2510_21991.md)
- [wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md](../../sources/blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md)
- [arXiv:2510.21991](https://arxiv.org/abs/2510.21991)
