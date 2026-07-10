---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2602.03511"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_cmr.md
summary: "非结构地形上，人形机器人的传感器会出错、模型也不准，观测噪声容易在闭环里被放大成失稳。CMR 的思路是：与其在原始观测空间里硬抗噪声，不如把观测映射到一个「收缩」潜空间——在那里相邻状态的扰动会随时间逐步收缩衰减。它用对比学习（InfoNCE）保住任务相关的语义结构（防止收缩过度把有用信息也压没了），同时用 Lipschitz 正则显式逼策略满足收缩条件，二者写成一个辅助损失直接加进 PPO；理论上还证明了：当潜动态满足收缩性（κ<1），策略性能因噪声的退化被一个与时间步长无关的上界 O(η/(1−κ)) 框住，而非标准分析里随时域指数爆炸。"
---

# CMR

**CMR: Contractive Mapping Embeddings for Robust Humanoid Locomotion on Unstructured Terrains** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：05_Locomotion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

非结构地形上，人形机器人的传感器会出错、模型也不准，观测噪声容易在闭环里被放大成失稳。CMR 的思路是：与其在原始观测空间里硬抗噪声，不如把观测映射到一个「收缩」潜空间——在那里相邻状态的扰动会随时间逐步收缩衰减。它用对比学习（InfoNCE）保住任务相关的语义结构（防止收缩过度把有用信息也压没了），同时用 Lipschitz 正则显式逼策略满足收缩条件，二者写成一个辅助损失直接加进 PPO；理论上还证明了：当潜动态满足收缩性（κ<1），策略性能因噪声的退化被一个与时间步长无关的上界 O(η/(1−κ)) 框住，而非标准分析里随时域指数爆炸。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| CMR | Contractive Mapping for Robustness | 本文方法名：收缩映射做鲁棒性 |
| Contractive Mapping | 收缩映射 | 把观测映到潜空间后，扰动随时间衰减而非放大 |
| Contraction Theory | 收缩理论 | 用收缩映射定理给「噪声→性能退化」一个紧上界 |
| InfoNCE | 对比学习损失 | 保留任务相关语义结构，防止收缩过度丢信息 |
| Lipschitz | 利普希茨约束 | 限制映射对输入的敏感度，显式逼出收缩性 |
| κ (kappa) | 收缩因子 | <1 时潜动态收缩；本文取 0.1 |
| HIM / LCP | Hybrid Internal Model / Lipschitz-Constrained Policies | 两个对比基线 |

## 为什么重要

- **轻量即插**：只是一个辅助损失项，不动 PPO 主体、不加独立估计模块，迁移成本极低，适合挂到现有 locomotion 训练框架上。
- **抗噪即抗未建模扰动**：把「观测噪声」当成一类一般扰动来收缩，对传感缺失/失真、轻度建模误差都有缓冲作用。
- **理论给安全感**：收缩界 `O(η/(1−κ))` 与时域无关，意味着长时域 rollout 下鲁棒性不随步数崩坏，对长距离穿越有意义。
- **限制**：收缩因子 κ、Lipschitz 权重 λ 需调，过强收缩可能压掉有用信息（靠 InfoNCE 平衡）；论文主要在仿真 + sim-to-sim 验证，真机表现待后续；超出训练噪声包络（α≫3）的极端情形仍可能退化。

## 解决什么问题

1. **非结构地形 = 观测不可靠**：碎石、踏脚石、平衡木等地形上，本体感知/外感知都可能含噪、缺失或失真，模型本身也不精确。 2. **噪声会在闭环里被放大**：标准分析中，一步动作误差 η 经过 H 步会按 `O(L_f^H)` 指数放大，鲁棒性随时域迅速崩坏。 3. **现有抗噪手段代价高 / 不通用**：要么靠大量域随机化与工程调参堆鲁棒性，要么单独设计估计模块；缺一个**轻量、可即插进现成 RL 管线**的通用表示层。

**目标**：用一个**几乎零额外开销的辅助损失**，让策略在含噪观测下保持鲁棒，并给出可证明的退化上界。

## 核心机制

1. **收缩映射做鲁棒表示**：首次把**收缩理论**引入人形抗噪运动——观测扰动在潜空间随时间衰减，而非闭环放大。
2. **可证明的退化上界**：证明潜动态收缩时，噪声导致的性能退化被 `O(η/(1−κ))` 框住，**与时域无关**，跳出 `O(L_f^H)` 指数爆炸。
3. **对比 + Lipschitz 双目标**：InfoNCE 保任务信息、Lipschitz 逼收缩，二者互补，避免「只收缩不保信息」或「只保信息不抗噪」。
4. **即插即用、近零开销**：写成辅助损失 `L_InfoNCE + λL_Lipschitz + L_PPO`，直接挂进现成深度 RL（PPO）管线。
5. **实证抗噪更强**：六类非结构地形、五档噪声（α 从 0.01 到 3）下，比 HIM / LCP / Naive PPO 走得明显更远，sim-to-sim 零样本迁移指令跟踪更准、能耗更低。

方法拆解（深读笔记小节）：核心：把观测映到「收缩」潜空间；两个目标缝在一起；总损失（即插即用）；网络与训练管线。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/CMR__Contractive_Mapping_Embeddings_for_Robust_Humanoid_Locomotion/CMR__Contractive_Mapping_Embeddings_for_Robust_Humanoid_Locomotion.html> |
| arXiv | <https://arxiv.org/abs/2602.03511> |
| 机构 | University of Southampton · Westlake University · Nanjing University |
| 作者 | Qixin Zeng, Hongyin Zhang, Shangke Lyu, Junxi Jin, Donglin Wang\*, Chao Huang\* |
| 发表 | 2026-02-03 (arXiv) |
| 源码 | 截至当前论文未给出公开仓库链接（以 arXiv 后续版本/作者主页为准） |
| 笔记阅读日期 | 2026-06-15 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_cmr.md](../../sources/papers/humanoid_pnb_cmr.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/CMR__Contractive_Mapping_Embeddings_for_Robust_Humanoid_Locomotion/CMR__Contractive_Mapping_Embeddings_for_Robust_Humanoid_Locomotion.html>
- 论文：<https://arxiv.org/abs/2602.03511>

## 推荐继续阅读

- [机器人论文阅读笔记：CMR](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/CMR__Contractive_Mapping_Embeddings_for_Robust_Humanoid_Locomotion/CMR__Contractive_Mapping_Embeddings_for_Robust_Humanoid_Locomotion.html)
