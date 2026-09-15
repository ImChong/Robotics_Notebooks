# 论文笔记｜从原理到代码万字解析 MoRE：如何让人形机器人既会越障，又能切换拟人步态？

> 来源归档（blog / 微信公众号）

- **标题：** 论文笔记｜从原理到代码万字解析 MoRE：如何让人形机器人既会越障，又能切换拟人步态？
- **类型：** blog
- **作者：** RobotsHub（微信公众号；文风与 [wechat_robotshub_ppo_locomotion_fundamentals.md](wechat_robotshub_ppo_locomotion_fundamentals.md) 同属「万字解析 / 原理到代码」系列）
- **原始链接：** https://mp.weixin.qq.com/s/nSLC5OQAagcYDT6HhSHxIQ
- **发表日期：** 2026-09-15（入库日；页面未稳定暴露 `publish_time`）
- **入库日期：** 2026-09-15
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **一句话说明：** 对 arXiv:2506.08840 / [TeleHuman/MoRE](https://github.com/TeleHuman/MoRE) 的 **原理→消融→官方代码导读** 长文；**不新建论文实体**，交叉补强既有 [paper-amp-survey-08-more](../../wiki/entities/paper-amp-survey-08-more.md) 与 [sources/repos/more.md](../repos/more.md)。
- **步骤 2.5（开源核查）：** 项目页 <https://more-humanoid.github.io/> 与 GitHub **已开源**（训练/推理/部署脚本 + Stage 2 `body_mask_data` 外链）；与入库日既有结论一致，无新增仓库。

## 核心摘录（归纳，非全文）

### 文内因果链（一页记忆）

> 先用深度视觉学会越障 → 再修改网络内部的动作草稿 → 多个修改器分别提出建议 → 门控根据步态和地形分配权重 → 动作裁判保证走法像参考动作 → 属性分数调节膝高或蹲身高度 → 相机/动力学随机化帮助迁移实机。

### 三个「看似直接」的失败方案 → MoRE 三答案

| 失败方案 | MoRE 回答 |
|----------|-----------|
| 从零同时学越障 + 多步态（MoRE-OS） | **两阶段**：先 locomotion，再叠风格 |
| 直接残差关节命令（MoRE-A） | **latent residual**：改 actor 隐特征再进预训练 head |
| 单修改网络承担多步态 | **MoE 残差专家** + softmax gate 加权混合 |

### 三类奖励分工（文内强调）

| 奖励 | 回答问题 |
|------|----------|
| 运动 $r^l$ | 能否稳定穿越、跟踪速度 |
| 风格 $r^s$ | 连续 5 步关节轨迹是否像当前步态参考（多判别器 AMP） |
| 步态属性 $r^g$ | 膝高/基座高等可设计数值（补参考动作缺陷） |

**Stage 2 不冻结 base actor** — 预训练 checkpoint 只是可靠起点，基础策略与残差模块 **联合 PPO 更新**（Q7）。

### 困难楼梯成功率（文内 Table 解读）

| 方法/步态 | 成功率 |
|-----------|--------|
| Blind | 0.218 |
| Base（仅 Stage 1） | 0.660 |
| MoRE Walk–Run | 0.682 |
| MoRE High-Knees | 0.903 |
| MoRE Squat | 0.816 |

文内结论：高抬腿与楼梯动力学更匹配，但 **不同障碍最优步态不同**（困难沟槽 Walk–Run 0.933 > High-Knees 0.904）；**步态仍由外部指令指定**，论文未做自动地形→步态选择。

### 训练 vs 部署模块（文内表）

| 模块 | 训练 | 实机 |
|------|------|------|
| 深度/本体/速度/步态指令 | ✓ | ✓ |
| base + 残差专家 + gate + action head | 更新 | 固定推理 |
| critic 特权（高程图等） | ✓ | ✗ |
| 三判别器 + LAFAN1 参考 | 训练信号 | ✗ |

### 论文 vs 官方代码差异（复现必记）

论文写 experts 与 gate 同读 actor 特征 + gait command；**开源实现** `actor_critic_resi_moe.py` 用 `actor_input[:, 3:]` **去掉前 3 维 gait one-hot** 再送 base actor 与三 expert，**gate 仍读完整 `actor_input`** — 步态指令主要经 gate 路由专家组合，而非直接进 expert 输入（文内 §官方源码对照）。

### 官方代码导读（调用链）

```
g1_16dof_moe_residual_config.py
  → moe_residual_on_policy_runner_multi.py
  → actor_critic_resi_moe.py
  → resi_moe_ppo_multi.py
  → amp_discriminator_multi.py
```

- **风格奖励路由：** `runner` 内 `disc_reward * gait_commands[:, idx]` 实现「当前步态只听对应判别器」。
- **判别器奖励形：** `amp_reward_coef * clamp(1 - 0.25*(D-1)^2, min=0)`（与论文一致）。

### 训练算力（论文 vs README）

| 阶段 | 论文 | README/配置建议 |
|------|------|-----------------|
| Stage 1 | 1×4090，~10k iter | **30k–50k**（建议 ≥40k），≥3000 env |
| Stage 2 | 4×4090，~20k iter | **40k**（前 30k 残差 + 后 10k body mask），≥6000 env |

### 证据边界（文内审慎读法）

- 真机 G1 **零微调**展示 0.4 m 沟槽、0.3 m 台阶、三阶 0.15 m 楼梯与组合地形。
- **未报告**实机重复次数、成功率分布、失败类型与端到端延迟；宜表述为「具备部署能力」，不宜过度断言统计可靠性。

## 对 wiki 的映射

- **复用实体：** [paper-amp-survey-08-more](../../wiki/entities/paper-amp-survey-08-more.md) — 增补代码导读、论文/代码差异、训练-部署对照与 FAQ 要点。
- **复用仓库归档：** [sources/repos/more.md](../repos/more.md) — 补关键源码路径表。
- **交叉：** [amp-reward.md](../../wiki/methods/amp-reward.md)、[terrain-adaptation.md](../../wiki/concepts/terrain-adaptation.md)、[locomotion.md](../../wiki/tasks/locomotion.md)、[unitree-g1.md](../../wiki/entities/unitree-g1.md)。

## 当前提炼状态

- [x] 公众号正文抓取（WebFetch）
- [x] 项目页/仓库开源核查（步骤 2.5）
- [x] 既有 MoRE 实体交叉补强（不重复造页）
