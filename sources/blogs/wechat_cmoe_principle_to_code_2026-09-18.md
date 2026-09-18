# 万字逐模块解读王兴兴收藏的项目 CMoE 如何使得 G1 准确适应不同地形

> 来源归档（blog / 微信公众号）

- **标题：** 万字逐模块解读王兴兴收藏的项目CMoE如何使得G1准确适应不同地形 |全文主线
- **类型：** blog
- **作者：** 微信公众号（原文页未稳定暴露账号名；文风为「万字逐模块解读 / 原理→代码」系列）
- **原始链接：** https://mp.weixin.qq.com/s/l6cy5nodRTfORY8SKwXXDw
- **发表日期：** 2026-09-18（入库日；页面未稳定暴露 `publish_time`）
- **入库日期：** 2026-09-18
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **原始抓取落盘：** [`sources/raw/wechat_cmoe_principle_to_code_2026-09-18.md`](../raw/wechat_cmoe_principle_to_code_2026-09-18.md)
- **一句话说明：** 对 [arXiv:2603.03067](https://arxiv.org/abs/2603.03067) / [Hoshi-No-Ai/CMoE](https://github.com/Hoshi-No-Ai/CMoE) 的 **原理→Fig.3→官方代码导读** 长文；**不新建论文实体**，交叉补强既有 [paper-cmoe](../../wiki/entities/paper-cmoe.md) 与 [sources/repos/cmoe.md](../repos/cmoe.md)。
- **步骤 2.5（开源核查）：** 项目页 <https://hoshi-no-ai.github.io/CMoE/> 与 GitHub **已开源**（`train.py`/`play.py` + `cmoe_ppo`）；无预训练 checkpoint；与入库日既有结论一致。

## 核心摘录（归纳，非全文）

### 文内因果链（一页记忆）

> Vanilla MoE 门控不随地形变化（lazy gating）→ 多专家难以分工 → CMoE 用 SwAV 把 gate 激活与高程 latent 钉到共同 prototype 空间 → Sinkhorn 防坍缩、交换预测对齐两侧 → 门控按地形重组权重 → PPO 在不同组合下学动作 → Expert 1 等涌现可干预验证的专长 → 单策略完成坡/楼梯/沟/栏/混合地形与 G1 真机部署。

### 两类常见方案 vs CMoE

| 路线 | 文内读法 |
|------|----------|
| 两阶段「单地形预训 + 蒸馏统一策略」 | 训练更长、二阶段易过拟合；CMoE **单阶段**八类地形同训 |
| Vanilla MoE | 专家多 ≠ 自动分工；门控激活 t-SNE **各地形混杂** |
| **CMoE** | 地形对比学习约束 gate；相似地形 → 相似专家组合 |

### 双 Estimator 分工

| 模块 | 输入 | 损失 | 部署 |
|------|------|------|------|
| **β-VAE 状态估计** | 本体历史 \(o^H\) | 显式体速 MSE + **下一帧** \(o_{t+1}\) 预测 MSE + KL | encoder 输出 \(\tilde v_t, z^H\) 进 policy；**decoder 仅训练用** |
| **地形 AE** | 高程图 | 高程重构 MSE（无 KL） | \(z^E\) + 原始 77 维高程进 policy |

文内强调：VAE **不是重构当前帧**，而是让 \(z^H\) 必须保留对 **下一时刻演化** 有用的动态信息；KL + 采样防止 latent 碎裂成互不相关的记忆孤岛。

### 157 维 `actor_input`（与官方代码一致）

| 部分 | 维 | 含义 |
|------|---:|------|
| 当前 observation | 45 | 本体、指令、上一动作 |
| VAE 显式输出 | 3 | 估计机体线速度 |
| VAE history latent | 16 | 动态预测隐状态 |
| elevation map | 77 | 原始局部高程 |
| terrain latent | 16 | 地形 AE 压缩 |
| **合计** | **157** | Gate 与 **全部 5 个 Expert** 共用 |

**Dense MoE：** 每步 **5 个 Expert 全部前向**，gate softmax 连续加权；不是 sparse top-1 路由。

### Gate / Actor / Critic 代码要点（`cmoe_actor_critic.py`）

- Gate：`Linear(157→128) → Softmax → 5 权重`；与 experts **同读** `actor_input`。
- 动作：\(\mu = \sum_i w_i \mu_i\)；探索噪声用 **CMoE 顶层共享** `self.std`，不再逐 expert 混合 std。
- Critic：`value_gate_weights = self.gate_weights.detach()` — value loss **不回传** gate，避免 gate 为「好拟合 value」而非地形路由而漂移（**代码事实**，非论文公式显式写出）。
- 对比分支：`gate_projector(gate_input)` vs `terrain_projector(height+latent2)` → 32 prototypes → Sinkhorn + SwAV 互预测。

### 训练 vs 部署

| 路径 | 训练 | 部署 |
|------|------|------|
| 双 estimator → 157 维 → gate + 5 experts | ✓ | ✓ |
| critic + privileged obs | ✓ | ✗ |
| SwAV 对比损失 | ✓ | ✗ |
| VAE decoder 下一帧预测 | 辅助头 | ✗ |

### 证据边界（文内审慎读法）

- 仅 **Expert 1** 有完整屏蔽实验（上楼受损、下楼仍可）；其余 4 专家未逐一枚举。
- **无** 地形 AE vs 对比损失独立消融；**无** 单共享 Critic vs 多 Critic 对照。
- t-SNE 只能说明相关性，不能证明高维聚类 = 可靠地形类。
- 真机为能力展示，**未报告** 重复次数、成功率分布、感知失效统计。
- Dense MoE **每步算满 5 expert**，证明分工价值，不涉及 sparse MoE 省算力。

## 对 wiki 的映射

- **复用实体：** [paper-cmoe](../../wiki/entities/paper-cmoe.md) — 增补 157 维输入表、VAE 下一帧预测、gate detach、dense MoE 与证据边界。
- **复用仓库归档：** [sources/repos/cmoe.md](../repos/cmoe.md) — 补 `cmoe_actor_critic.py` 关键路径与 `actor_input` 组装顺序。
- **交叉：** [terrain-adaptation.md](../../wiki/concepts/terrain-adaptation.md)、[stair-obstacle-perceptive-locomotion.md](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[unitree-g1.md](../../wiki/entities/unitree-g1.md)、[paper-amp-survey-08-more.md](../../wiki/entities/paper-amp-survey-08-more.md)（两阶段 MoE 对照）

## 当前提炼状态

- [x] 公众号正文抓取（WebFetch）
- [x] 项目页/仓库开源核查（步骤 2.5）
- [x] 既有 CMoE 实体交叉补强（不重复造页）
