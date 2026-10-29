# 借着supermimic复习：Model base，可微世界模型，BPTT

> 来源归档（blog / 微信公众号）

- **标题：** 借着supermimic复习：Model base，可微世界模型，BPTT
- **类型：** blog
- **作者：** senlanke / 具身运控lab（微信公众号，「论文笔记」合集）
- **原始链接：** https://mp.weixin.qq.com/s/NYg_AGSlVcK6jxxipuK3vA
- **发表日期：** 2026-10-29
- **入库日期：** 2026-10-29
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA，AGENTS.md「兜底」流程），正文 10975 字符
- **一句话说明：** 以 SuperMimic 为引子，系统复习 Model-Free vs Model-Based RL、动力学模型输入输出、交替训练（MBPO / Dreamer）、可微模型 + BPTT、「预测值准≠梯度准」，以及机器人 RL 为何仍普遍用 PPO。
- **说明：** 文章是概念讲解而非项目清单；下表把正文**点名的每个算法 / 系统 / 工具**各映射到唯一详情节点（无重复新建）。公式在抓取纯文本中丢失，技术结论以正文文字为准。

## 文章结构

1. Model-Free 与 Model-Based 的定义、流程与对比
2. 动力学模型的输入输出（状态 / 动作 / 动力学参数，隐变量，Policy 与模型输入可不同）
3. 交替训练循环：收集真实轨迹 → 训练环境模型 → 用模型训练策略（MBPO + SAC；Dreamer）；模型误差累积
4. 让梯度穿过模型：反向传播、模型可微性、一步链式法则
5. BPTT：多步预测即沿时间展开；PPO 与可微 MBRL 的区别
6. 预测值准，不等于梯度准（误差为 0 而梯度反向的反例）
7. 为什么机器人 RL 仍普遍用 PPO（Isaac Lab 提供大量低成本交互；仿真器不提供状态转移梯度）

## 点名项目 → 详情节点

| 文中提到 | 文中角色 | 详情节点 | 状态 |
|----------|----------|----------|------|
| SuperMimic | 标题引子：联合训练策略与可微世界模型 | [SuperMimic](../../wiki/entities/paper-supermimic.md) | 复用 |
| PPO | Model-Free 代表；机器人 RL 主流 | [PPO](../../wiki/methods/ppo.md) | 复用 |
| SAC | Model-Free 代表；MBPO 的策略更新器 | [SAC](../../wiki/methods/sac.md) | 复用 |
| TD3 | Model-Free 代表算法 | [TD3](../../wiki/methods/td3.md) | **新建** |
| PETS | Model-Based 代表算法 | [PETS](../../wiki/entities/paper-pets-probabilistic-dynamics-mpc.md) | 复用 |
| MBPO | 短期虚拟轨迹 + SAC 的完整流程 | [MBPO](../../wiki/methods/mbpo.md) | **新建** |
| Dreamer | 世界模型虚拟轨迹 + 梯度穿过模型 | [Dreamer](../../wiki/entities/paper-dreamer-latent-imagination.md) | 复用 |
| MPC | 基于规划的 Model-Based 用法 | [模型预测控制](../../wiki/methods/model-predictive-control.md) | 复用 |
| RSSM | 可自动微分的 latent 动力学模型 | [PlaNet（RSSM 出处）](../../wiki/entities/paper-planet-latent-dynamics.md) | 复用 |
| BPTT | 梯度沿时间步反传 | [BPTT](../../wiki/concepts/bptt.md) | **新建** |
| Isaac Lab | PPO 所用大规模并行仿真 | [Isaac Lab](../../wiki/entities/isaac-lab.md) | 复用 |
| CTS（文末上一篇链接） | 仅导航链接，非正文论述 | [CTS](../../wiki/entities/paper-cts-concurrent-teacher-student-locomotion.md) | 复用 |
| Model-Free / Model-Based RL | 全文主线对比 | [Model-Based RL](../../wiki/methods/model-based-rl.md)、[对比页](../../wiki/comparisons/model-based-vs-model-free.md) | 复用 |

## 对 wiki 的映射

- 新建 [TD3](../../wiki/methods/td3.md)、[MBPO](../../wiki/methods/mbpo.md)、[BPTT](../../wiki/concepts/bptt.md)；其余全部复用已有节点并补充反向链接。
