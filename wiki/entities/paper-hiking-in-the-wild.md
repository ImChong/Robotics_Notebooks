---
type: entity
tags: [paper, humanoid, rl, motion-control, perceptive-locomotion, depth, parkour, tsinghua, amp]
status: complete
updated: 2026-09-04
arxiv: "2601.07718"
venue: arXiv
summary: "Hiking in the Wild（arXiv:2601.07718）单阶段 E2E 深度+本体 RL，用地形边缘与足端体积点软约束实现野外 2.5 m/s 感知跑酷，含 AMP-style 自然性项与开源栈。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ../methods/amp-reward.md
  - ./paper-amp-survey-08-more.md
  - ./paper-deep-whole-body-parkour.md
  - ./paper-parkourformer.md
  - ./paper-cref.md
  - ./paper-wm-loco.md
sources:
  - ../../sources/papers/hiking_in_the_wild_arxiv_2601_07718.md
  - ../../sources/papers/humanoid_rl_stack_24_hiking_in_the_wild_a_scalable_perceptive_parkour.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/humanoid_amp_survey_09_hiking_in_the_wild_a_scalable_perceptive_parkour.md
  - ../../sources/papers/humanoid_amp_survey_19_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md
---

# Hiking in the Wild：可扩展感知跑酷框架

**Hiking in the Wild**（arXiv:2601.07718）由清华大学交叉信息研究院等提出，在 [42 篇 RL 身体系统栈](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) 为 **24/42**（03 感知式高动态运动），在 [AMP 运动先验专题](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w) 为 **09/19**（**02 人形走跑**）。策展强调：**不是典型 AMP 论文**，但应放在走跑线——它把 **深度感知 E2E locomotion** 与 **AMP-style 自然性奖励** 放在同一单阶段策略里。

## 一句话定义

**单阶段端到端 RL：原始深度图与本体历史直接映射关节目标，配合地形边缘检测与足端体积点软约束、Flat Patch Sampling 导航命令与 MoE 策略，在野外真机达 2.5 m/s 且无外部定位。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| E2E | End-to-End | 深度感知不经显式建图中间层 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 总奖励中含 AMP-style 自然性项 |
| MoE | Mixture-of-Experts | 策略网络专家混合，消融验证有效 |
| PD | Proportional–Derivative | 关节目标 + 底层 PD 执行 |
| Sim2Real | Simulation to Real | Warp 深度仿真与真机噪声双向对齐 |

## 为什么重要

- **野外可持续穿越：** 相对 PHP、Deep Whole-body Parkour 等，更强调 **楼梯、沟壑、高台、坡地、边缘密集区** 的**持续通过**而非单障碍特技。
- **开源可复现：** 训练与部署代码开源（项目页）；相对同类工作「少开源」的痛点。
- **AMP 纵深交叉：** 总奖励 $R=r_{\mathrm{task}}+r_{\mathrm{reg}}+r_{\mathrm{safe}}+r_{\mathrm{amp}}$；消融去掉 AMP 在 Small Box 上 **0% vs 99.09%**——说明感知跑酷仍需要 **运动先验/自然性** 正则。
- **与 MoRE 姊妹：** [MoRE #08](./paper-amp-survey-08-more.md) 两阶段「先穿越再 AMP」；Hiking **单阶段** 深度 E2E + AMP 项并联。

## 流程总览

```mermaid
flowchart TB
  subgraph obs [观测]
    prop["本体历史 h 步"]
    depth["深度历史 H_t<br/>Warp 仿真"]
    critic["非对称 critic<br/>特权线速度等"]
    prop --> pol
    depth --> pol
  end
  subgraph pol [策略 MoE]
    pol["π → 29 维关节目标"]
  end
  subgraph reward [奖励]
    task["r_task<br/>Flat Patch 速度命令"]
    safe["r_safe<br/>边缘+足端体积点"]
    amp["r_amp<br/>AMP-style 自然性"]
    reg["r_reg"]
  end
  subgraph deploy [真机]
    cam["机载深度 60 Hz"]
    wild["野外楼梯/坡地/草地"]
    cam --> pol --> wild
  end
  obs --> pol
  task --> pol
  safe --> pol
  amp --> pol
  reg --> pol
```

## 核心机制（归纳）

### 1）感知与 sim2real

- **观测：** 本体历史 + 深度历史；**非对称 critic** 含真值线速度等。
- **深度：** NVIDIA **Warp** 并行 ray-cast；$\mathcal{F}_{sim}$ / $\mathcal{F}_{real}$ **双向对齐**噪声，零样本 sim2real。
- **动作：** 29 维关节目标 + PD（增益沿用 BeyondMimic）。

### 2）安全与防 hack

- **Terrain Edge Detection：** 任意 trimesh 自动提取边缘。
- **Foot Volume Points：** 足端穿透边缘惩罚 → 学「脚踩实面中心」。
- **Flat Patch Sampling：** 网格上采可达平面块生成相对速度命令，避免原地转圈 reward hacking。

### 3）训练与真机

| 项目 | 内容 |
|------|------|
| 策略 | 含 **MoE**；单阶段 E2E |
| 真机 | 机载前向深度 60 Hz；**无外部定位** |
| 速度 | 野外最高 **2.5 m/s** |
| 开源 | 项目页代码与文档 |

## 结论

**Hiking in the Wild 主张感知跑酷不必分阶段：深度 E2E 与 AMP-style 自然性项可以并联进同一个单阶段策略；换来的是流程简化，代价是奖励项之间的配平成了系统关键路径。**

- 最硬的证据是去 AMP 消融在 Small Box 上 **0% vs 99.09%**——自然性项不是锦上添花，在窄障碍上它直接决定能否通过。
- 能在野外跑起来靠的是三件防 hack 设计：**地形边缘检测**、**足端体积点** 惩罚（学会踩实面中心）、**Flat Patch Sampling** 生成可达平面块速度命令（避免原地转圈刷奖励）。
- Sim2real 走 **Warp 并行 ray-cast + $\mathcal{F}_{sim}$/$\mathcal{F}_{real}$ 双向噪声对齐**：机载深度 60 Hz、**无外部定位**、野外 **2.5 m/s**——路线是轻量 E2E，而非建图/SLAM。
- 定位差异在任务口径：强调楼梯、沟壑、高台、坡地等 **边缘密集区的持续通过**，而不是单障碍特技；MoE 与深度历史等模块在 10 类地形消融中均有贡献。
- 与 [MoRE #08](./paper-amp-survey-08-more.md) 对照：那边是 **两阶段 + gait command + 多判别器**，本页是 **单阶段 + 单一 AMP-style 项 + Flat Patch 命令生成**，二者不是简化关系。

## 常见误区

1. **纯感知不需要运动先验：** 消融 **去 AMP** 在 Small Box **成功率归零**——自然性项对窄障碍关键。
2. **= MoRE 简化版：** MoRE **两阶段** + gait command + 多判别器；Hiking **单阶段** + 单一 AMP-style 项 + 不同命令生成（Flat Patch）。
3. **依赖 SLAM/建图：** **E2E 深度**，强调无定位漂移的轻量方案。
4. **仅仿真跑酷：** 项目页有野外真机视频（楼梯、坡地、草地、离散障碍）。

## 实验与评测

- **仿真 10 类地形：** 完整系统 vs 去 MoE / 去深度历史 / 去 pose-based 命令 / 去 AMP；每项消融降成功率。
- **真机：** 2.5 m/s 野外穿越；无外部定位。
- **双索引：** 42 篇栈 #24 + AMP 专题 #09。

## 与其他页面的关系

- 姊妹感知跑酷：[MoRE #08](./paper-amp-survey-08-more.md)、[Deep Whole-body Parkour](./paper-deep-whole-body-parkour.md)
- 同族 MuJoCo、改未来监督：[ParkourFormer](./paper-parkourformer.md) — Instinct 九类课 + query 历史 / 未来两步 AMP；代码 Coming Soon
- 无 AMP、无建图的单阶段 raw 深度：[CReF](./paper-cref.md) — 交叉注意 + 落脚奖励；X2 Ultra 实验室课
- 复用足端体积点 / 边缘惩罚、加上 RSSM 共训：[WM-LOCO](./paper-wm-loco.md) — G1 沟/踏石上匹配 PPO 归零
- 任务：[stair-obstacle-perceptive-locomotion.md](../tasks/stair-obstacle-perceptive-locomotion.md)
- RL 栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 专题：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)（#09/19）

## 参考来源

- [Hiking in the Wild（arXiv:2601.07718）](../../sources/papers/hiking_in_the_wild_arxiv_2601_07718.md)
- [humanoid_rl_stack_24_hiking_in_the_wild_a_scalable_perceptive_parkour.md](../../sources/papers/humanoid_rl_stack_24_hiking_in_the_wild_a_scalable_perceptive_parkour.md)
- [humanoid_amp_survey_09_hiking_in_the_wild_a_scalable_perceptive_parkour.md](../../sources/papers/humanoid_amp_survey_09_hiking_in_the_wild_a_scalable_perceptive_parkour.md)
- [humanoid_amp_survey_19_catalog.md](../../sources/papers/humanoid_amp_survey_19_catalog.md)
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)

## 推荐继续阅读

- [项目页](https://project-instinct.github.io/hiking-in-the-wild) — 视频与代码
- [arXiv:2601.07718](https://arxiv.org/abs/2601.07718)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [SD-AMP 深读页](./paper-unified-walk-run-recovery-sdamp.md) — 另一路「自然性+统一策略」对照
