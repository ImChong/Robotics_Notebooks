---
type: method
tags: [humanoid, tactile-sensing, visuo-tactile, imitation-learning, behavior-cloning, loco-manipulation, transformer]
status: complete
updated: 2026-04-29
related:
  - ./imitation-learning.md
  - ./bc-with-transformer.md
  - ./action-chunking.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../concepts/contact-rich-manipulation.md
  - ../tasks/loco-manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/humanoid_touch_dream.md
summary: "Humanoid Transformer with Touch Dreaming (HTD) 是一种面向人形机器人接触丰富型移动操作的多模态模仿学习策略，通过未来手部力和触觉 latent 预测，让 Transformer 在行为克隆中学到接触感知表征。"
---

# Humanoid Transformer with Touch Dreaming (HTD)

**HTD** 是一种面向人形机器人 dexterous loco-manipulation 的多模态行为克隆方法。它把触觉从“附加传感器输入”提升为训练目标：策略不仅预测 action chunks，还要预测未来手部关节力和未来触觉 latent，从而逼迫共享 Transformer trunk 学到接触动态。

## 一句话定义

HTD = 多视角视觉 + 本体感受 + 手部力 + 分布式触觉输入的 Transformer policy，再用 **Touch Dreaming** 预测未来接触信号来正则化表示学习。

## 为什么重要

人形机器人接触丰富型操作不是固定机械臂操作的简单放大版。手部接触会通过躯干姿态、底盘速度、足端支撑和双手协调影响全身稳定性；如果策略只学图像到动作，常会在插入、折叠、工具使用、端茶行走这类任务中丢失接触阶段的关键状态。

HTD 的关键价值在于两点：

1. **触觉不是简单拼接输入。** 论文的消融显示，仅给 ACT 加触觉输入并不稳定；未来触觉预测才是让策略真正使用触觉的训练压力。
2. **触觉预测放在 latent space。** Raw tactile array 高维、稀疏且噪声大，直接回归容易学到传感器噪声；EMA teacher 产生的 tactile latent 更适合作为稳定监督。

## 系统结构

HTD 依赖一个完整的人形操作数据闭环：

1. **Lower-Body Controller (LBC)：** 在 IsaacLab 中训练 RL teacher，再用 DAgger 蒸馏到只依赖真机可观测本体信息的 student。LBC 跟踪底盘速度、躯干姿态和高度命令，是操作时的稳定性骨架。
2. **VR 遥操作采集：** 操作者的头、腕、手信号被映射为躯干命令、腕部 6D pose、灵巧手 retargeting target，摇杆提供底盘速度。
3. **多模态 demonstration：** 数据同步包含头部/腕部 RGB、本体感受、手部关节力、双手触觉，以及 torso、end-effector、velocity、hand action targets。
4. **HTD policy：** 多模态 tokenizer 将各输入压成 tokens，encoder-decoder Transformer 融合后，由 action experts 解码动作，由 dream experts 解码未来接触信号。

## 主要技术路线

HTD 的路线不是“更大的端到端网络”，而是把稳定执行、遥操作采集和触觉预测式学习拆成三层：

1. **稳定性先外包给 LBC。** 下肢和躯干由 RL lower-body controller 承担，策略学习主要处理上身、手部和速度命令，降低全身行为克隆的动力学负担。
2. **示范数据必须包含接触。** VR teleoperation 同步采集视觉、本体、手部力和触觉，保证接触阶段不是事后补标签。
3. **触觉通过预测目标进入表示。** 训练时预测未来 hand force 与 tactile latent，让 Transformer trunk 对接触变化敏感；推理时只保留动作输出，避免增加部署链路复杂度。

## Touch Dreaming 机制

Touch Dreaming 是训练期辅助目标，部署时不参与控制。

| 分支 | 预测目标 | 监督方式 | 作用 |
|------|----------|----------|------|
| Action experts | 短 horizon action chunks | 行为克隆 loss | 输出实际控制目标 |
| Force dream expert | 未来手部关节力 | Smooth L1 | 学习接触强度和时序 |
| Tactile latent dream expert | 未来触觉 latent | EMA tactile tokenizer + cosine / magnitude loss | 学习空间接触结构，降低 raw tactile 噪声影响 |

EMA tactile tokenizer 的作用类似慢速 teacher：它不反传梯度，只提供稳定 latent target，避免 student tokenizer 和 tactile detokenizer 共同塌缩到无信息表示。

## 实验结论

论文在五个真实接触丰富型任务上测试 HTD：Insert-T、Book Organization、Towel Folding、Cat Litter Scooping、Tea Serving。任务覆盖紧公差插入、薄物体推抓、可变形物体折叠、蹲下工具操作和双手端杯移动操作。

核心结论：

- HTD 在五个任务上都超过 decoder-only ACT baseline。
- 相比更强 ACT variant，HTD 平均成功率提升 30.0 个百分点，约 90.9% relative improvement。
- 相比 raw tactile prediction，latent tactile dreaming 在成功率上有约 30% relative gain。
- 简单加入触觉输入并不保证收益；显式预测未来接触信号才让策略形成接触感知表征。

## 常见误区

- **误区 1：HTD 是一个在线触觉世界模型。**  
  不是。Dream heads 只在训练中提供辅助 loss，部署时只用 action experts。

- **误区 2：触觉越原始越好。**  
  对高维稀疏触觉阵列来说，raw prediction 容易追噪声；latent prediction 更稳定。

- **误区 3：有触觉输入就等于会用触觉。**  
  论文消融表明，被动加入触觉观测不一定提升控制；需要预测式目标让网络学习接触时序。

## 与其他页面的关系

- [视触觉融合](../concepts/visuo-tactile-fusion.md)：HTD 是 attention/token 化路线上的具体实例，但它额外强调未来触觉预测。
- [触觉感知](../concepts/tactile-sensing.md)：HTD 使用双手分布式 tactile sensing 和手部关节力作为接触观测。
- [Loco-Manipulation](../tasks/loco-manipulation.md)：HTD 的任务不是固定底座操作，而是需要下肢稳定、躯干姿态和双手操作同时协调。
- [Imitation Learning](./imitation-learning.md)：HTD 本质上是单阶段行为克隆，只是加入 touch dreaming 辅助目标。
- [Action Chunking](./action-chunking.md)：HTD 继承 action chunking，用短 horizon 目标提升时序平滑性。

## 关联页面

- [Imitation Learning](./imitation-learning.md)
- [Behavior Cloning with Transformer](./bc-with-transformer.md)
- [Action Chunking](./action-chunking.md)
- [Tactile Sensing](../concepts/tactile-sensing.md)
- [Visuo-Tactile Fusion](../concepts/visuo-tactile-fusion.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [Teleoperation](../tasks/teleoperation.md)

## 推荐继续阅读

- [HTD 项目主页](https://humanoid-touch-dream.github.io/)
- [arXiv:2604.13015](https://arxiv.org/abs/2604.13015)
- [ViTacFormer](https://arxiv.org/abs/2506.15953) — 视触觉表征学习相关工作
- [ACT / Action Chunking](./action-chunking.md) — HTD 对比和继承的行为克隆基线

## 参考来源

- [sources/papers/humanoid_touch_dream.md](../../sources/papers/humanoid_touch_dream.md)
