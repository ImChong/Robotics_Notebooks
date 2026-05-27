---
type: comparison
tags: [amp, motion-prior, rl, humanoid, imitation-learning, comparison]
status: complete
updated: 2026-05-21
summary: "AMP / ADD / SMP 三种对抗式运动先验变体：判别器形式、多目标解耦与模块化 reward model 的选型对比。"
sources:
  - ../../sources/papers/amp.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md
related:
  - ../methods/amp-reward.md
  - ../methods/add.md
  - ../methods/smp.md
  - ../methods/deepmimic.md
  - ../overview/humanoid-amp-motion-prior-survey.md
---

# AMP vs ADD vs SMP：运动先验变体对比

**背景**：在人形 RL 中，显式 tracking 奖励容易带来「能完成任务但不像人」的步态。[AMP](../methods/amp-reward.md) 用判别器学习参考动作分布；社区在此基础上演化出 [ADD](../methods/add.md)（对抗差分、减轻多目标手调）与 [SMP](../methods/smp.md)（可复用 motion prior reward model）。三者共享「用对抗信号约束风格」的哲学，但在**目标解耦、工程模块化与调参成本**上取舍不同。

> **一句话区分**：AMP 是「标准判别器先验」；ADD 是「用差分对抗减轻 reward 纠缠」；SMP 是「把先验拆成独立可插拔模块」。

---

## 核心维度对比

| 维度 | **AMP** | **ADD** | **SMP** |
|------|---------|---------|---------|
| **核心机制** | 判别器区分参考 vs 策略 rollout | 对抗**差分**信号，弱化与任务 reward 的手动平衡 | 运动先验拆为独立 reward model |
| **典型输入** | 状态（+ 可选历史）片段 | 与 AMP 类似，强调差分构造 | 参考库 + 策略特征 |
| **与任务 reward** | 常需仔细调 style/task 权重 | 设计上减轻多目标耦合 | 模块化叠加，便于 ablation |
| **实现复杂度** | 中（判别器 + PPO） | 中高（差分对抗结构） | 中高（reward model 管线） |
| **调参痛点** | style 权重、判别器容量 | 差分项与任务项边界 | reward model 训练稳定性 |
| **最佳场景** | 通用自然步态、行走舞蹈 | 多任务 reward 已很复杂 | 需要频繁换 prior / 做模块实验 |

---

## 什么时候选哪一个

**选 AMP**：已有成熟 AMP 基线（如 MimicKit / ProtoMotions），需要最快接入「自然风格」约束。

**选 ADD**：任务 reward 项很多（速度、平衡、接触、末端），不想再手工调一大组 style 权重。

**选 SMP**：实验矩阵要求**频繁替换或组合** motion prior，希望 prior 与 policy 训练解耦。G1 + mjlab 工程入口见 [SMP on G1（mjlab 复现）](../entities/smp-g1-mjlab.md)（与 [AMP_mjlab](../entities/amp-mjlab.md) 对照）。

---

## 常见误判

1. **三者可互换**：实现细节与超参尺度不同，不能直接复制 AMP 权重到 ADD/SMP。
2. **有 prior 就不需要 tracking**：复杂动作仍建议保留适度 tracking 或终止条件，prior 只修分布不保证任务。
3. **判别器越大越好**：过大判别器会导致 mode collapse 或策略梯度噪声放大。

---

## 结论

- **默认起点**：[AMP](../methods/amp-reward.md)
- **多目标 reward 已臃肿** → 评估 [ADD](../methods/add.md)
- **需要模块化 prior 实验** → 评估 [SMP](../methods/smp.md)

完整栈内位置见 [人形运动跟踪方法选型指南](../queries/humanoid-motion-tracking-method-selection.md)。

---

## 参考来源

- [AMP 论文摘要](../../sources/papers/amp.md)
- [具身智能研究室：人形 AMP 先验综述](../../sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)

## 关联页面

- [AMP & HumanX](../methods/amp-reward.md)
- [ADD](../methods/add.md)
- [SMP](../methods/smp.md)
- [SMP on G1（mjlab）](../entities/smp-g1-mjlab.md)
- [DeepMimic](../methods/deepmimic.md)
- [人形运动跟踪方法选型](../queries/humanoid-motion-tracking-method-selection.md)
- [人形 AMP 先验综述](../overview/humanoid-amp-motion-prior-survey.md)
