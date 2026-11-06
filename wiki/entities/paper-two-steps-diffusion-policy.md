---
type: entity
project_id: two-steps-diffusion-policy
tags:
  - paper
  - diffusion-policy
  - imitation-learning
  - inference-efficiency
status: complete
updated: 2026-11-05
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

**Two-Steps Diffusion Policy**（[arXiv:2510.21991](https://arxiv.org/abs/2510.21991)）是 [新机器视觉《2026年了，主流具身智能算法有哪些？》](https://mp.weixin.qq.com/s/aIYDRlrluZIsmj3fmMsXIg) 点名的算法之一。本页以 arXiv 摘要页与全文（v1 HTML）为一手资料，文章说法放在「文章说法核对」。

## 一句话定义

**Two-Steps Diffusion Policy 针对具身任务动作分布低维、结构化的特点定制去噪过程，使扩散策略仅需少量神经函数评估（NFE）即可有效运行，并提出 population-based 的 genetic denoising 提升性能与稳定性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| NFE | Neural Function Evaluation | 去噪网络前向调用次数，衡量推理开销 |
| GDP | Genetic Diffusion Policy | 本文的种群式（遗传）去噪采样策略 |
| DDPM | Denoising Diffusion Probabilistic Model | 带噪声注入的随机去噪采样器 |
| DDIM | Denoising Diffusion Implicit Model | 确定性去噪采样器 |
| EDM | Elucidating the Design space of diffusion Models（Karras 等） | 另一套扩散训练与采样框架，本文对照之一 |
| OoD | Out-of-Distribution | 分布外；本文指去噪中间态偏离训练分布 |

## 为什么重要

- 在文章的 2026 算法栈里属于：扩散策略 / 模仿学习层；总览见 [2026 具身智能算法栈：VLA、扩散策略、强化学习与世界模型（一手核对版）](../overview/embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md)。

## 一手资料（arXiv 摘要页）

- **标题 / 编号：** [Two-Steps Diffusion Policy](https://arxiv.org/abs/2510.21991)（提交 2025-10-24；NeurIPS 2025）
- **作者：** Clemente, Brunswic, Yang, Zhao, Khalil 等（arXiv 摘要页前 5 位）
- **摘要要点：** 摘要称：扩散策略在模仿专家示范的机器人操作上表现出色，但许多推理策略直接从视觉任务迁移而来；针对动作分布低维、结构化的特点定制去噪，可让扩散策略在少至 5 次 NFE 下有效运行；在此基础上提出 genetic denoising，选择分布外风险低的去噪轨迹；方法仅用 2 次 NFE 就解决困难任务且性能提升或持平；在 D4RL（Adroit Hand）与 Robomimic 的 14 项操作任务上、超过 200 万次评测中持续优于标准扩散策略，最高约 20% 性能提升。

## 方法（§2–3）

- **问题诊断**：标准 DDPM 推理每步把 x̂_0 截断（clip）到 [−1, 1]^d。去噪早期 x̂_0 大多被压到超立方体角上，中间态偏离训练分布，噪声预测误差变大；图 3 显示截断频率越高、最终回报越低（§2.2）。
- **少步 + 弱噪声**：论文认为动作分布低维、结构简单（Adroit 动作维度 24–30，估计内在维度约 11，附录 B），推理时可偏向「利用」：减少去噪步数、从 t < T 开始去噪、把注入噪声缩放系数 γ 调小（§2.3–2.4）。这些都不需要重训模型。
- **遗传去噪（GDP）**：同时采样一个种群，每步去噪前给每个样本算适应度分数，按多项分布选出一半并复制补满，再做一步去噪（算法 1）。分数有两类：基于噪声预测范数的 stein 分数与基于 x̂_0 被截断量的 clip 分数，都用来衡量样本有多「分布外」（式 4）。补充材料图 9–11 称 stein 分数是最好的简单启发式，clip 分数会删掉数值大的模态、扭曲采样分布。算法不含交叉与变异。
- **两步推理配置**：GDP 加时间步调整（最大步 90、最小步 20），γ=0.2（§4.2「Shortcut Models」）；补充材料的视频说明里两步配置写的是最大步 80，两处不一致，论文未解释。

## 评测（§4）

- **设置**：Adroit Hand 四任务（Hammer / Relocate / Pen / Door，每配置 100 个种子）与 Robomimic（Lift / Can / Square / Transport / ToolHang，PH / MH 数据，每配置 500 个种子）；所有方法用官方 Diffusion Policy 的 65M UNet，DDPM / GDP / DDIM 共用同一 DDPM 检查点，Shortcut 单独训练（每任务–时域 10 次取最好）；扫描步数 1–100、γ 0–1、动作时域 24–200（§4.1）。
- **Adroit（表 1）**：2 步时标准 DDPM 几乎失效（Hammer 0.00、Relocate 0.01）；GDP 2 步为 Hammer 1.00 / Relocate 0.98 / Pen 0.91 / Door 1.00，Shortcut 2 步为 0.88 / 1.00 / 0.81 / 0.94，100 步标准 DDPM（γ=1）为 0.68 / 0.69 / 0.88 / 0.87。只调时间表与 γ=0.2、不用 GDP 的 2 步变体已达 0.98 / 0.92 / 0.89 / 1.00。
- **Robomimic（表 2）**：各方法多在 2 个标准差内，**表注明确写 GDP 未能在「调整时间表」的变体之上进一步改进基础策略**；论文推测 Robomimic 的瓶颈在条件编码而非去噪过程（§6）。
- **步数与噪声（§4.2、图 5）**：截断频率随步数减少而下降，最佳步数出现在 n=4 之前；降低 γ 能让同一个 UNet 从约 75% 成功率提升到完全解决任务；中等动作时域比短 / 全长时域更难。
- **开销（表 3，RTX 3080）**：种群批量为一次前向；种群 8 / 16 / 32 / 64 的总时间分别是单样本 DDPM 的 1.08 / 1.20 / 1.50 / 1.98 倍。所以「2 NFE」指 2 次批量前向，不等于 2 次单样本前向的耗时。
- 摘要与 §1 称共 14 项任务，但表 1–2 列出的是 4 个 Adroit 任务与 9 个 Robomimic 任务–数据组合，计数方式论文未说明。
- 分数为论文的归一化成功率，只适用于其检查点、种子数与扫描网格，不能与其他论文直接横比。

## 与其他工作对比

- **[Diffusion Policy](./paper-diffusion-policy.md)（DDPM 采样）**：本文直接在其官方实现与检查点上改推理过程，不改训练与架构（§1、§4.1）；参见 [扩散策略方法页](../methods/diffusion-policy.md)。
- **DDIM**：Diffusion Policy 原文用 DDIM 减少步数，本文实测 DDIM 在 Adroit 上表现不佳（如 5 步 Relocate 0.38，表 1），解释为早期大步确定性更新面对分布外中间态不可靠（§2.4）。
- **EDM**：需从头重训并调参才有竞争力，且跨动作时域不稳定（从解决任务到成功率不足 50%，§2.4、图 5c）。
- **Shortcut 模型 / 一致性蒸馏**：Shortcut 是唯一能 1 步采样的方法，但需单独训练，表 1 中 1 步为 0.83 / 0.93 / 0.74 / 0.89；论文把它视为自蒸馏的一致性模型（§5）。本文路线是对现成模型免训练加速。
- **并行采样与群体方法**：论文把并行采样的平均操作类比为「交叉」，并把 GDP 放在群体元启发式与 SDE 求解器改进的脉络里（§5），但没有与这些方法做实测对比。

## 结论

论文主张：机器人动作分布低维、结构简单，图像生成里的推理习惯（多步、强噪声注入、DDIM 加速）不能照搬；少步、弱噪声与遗传去噪能让现成扩散策略在 2 次批量前向下保持或提升性能（§7）。

1. 提升主要来自 Adroit；Robomimic 上 GDP 没有超过只调时间表的变体（表 2 表注）。
2. 只调时间表与 γ 的免训练改动已贡献大部分收益，GDP 的额外增益有限但更稳。
3. 种群带来最多约 2 倍的单步耗时（表 3）；理论分析仍属初步，作者列为未来工作（§6）。全部实验在仿真中完成，未见真机评测。

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
