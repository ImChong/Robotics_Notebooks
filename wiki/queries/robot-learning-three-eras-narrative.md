---
type: query
tags: [history, embodied-ai, vla, imitation-learning, reinforcement-learning, scaling, google-deepmind, ted-xiao]
status: complete
updated: 2026-05-10
summary: "把『存在性证明 / 基础模型 / Scaling』当作阅读透镜：串联 QT-Opt→SayCan/RT/DIAL/OXE→Gemini 与社区评测，事实以论文与官方发布为准；组织叙事单独标注来源。"
sources:
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
  - ../../sources/papers/rl_foundation_models.md
related:
  - ../concepts/foundation-policy.md
  - ../concepts/deep-rl-game-milestones.md
  - ../concepts/open-x-embodiment.md
  - ../methods/vla.md
  - ../methods/qt-opt.md
  - ../methods/bc-z.md
  - ../methods/mt-opt.md
  - ../methods/learning-from-play-lmp.md
  - ../methods/cyclegan-sim2real.md
  - ../methods/saycan.md
  - ../methods/robotics-transformer-rt-series.md
  - ../methods/dial-instruction-augmentation.md
  - ../methods/octo-model.md
  - ../methods/roboarena.md
  - ../entities/gemini-robotics.md
  - ../entities/generalist-ai-robotics.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../concepts/embodied-scaling-laws.md
---

> **Query 产物**：本页由以下问题触发：「围绕 Ted Xiao 访谈编译稿中出现的机器人学习话题，能否用一手文献串联成可维护的知识索引？」  
> 综合来源：[Foundation Policy](../concepts/foundation-policy.md)、[VLA](../methods/vla.md)、[Imitation Learning](../methods/imitation-learning.md)、[Reinforcement Learning](../methods/reinforcement-learning.md)、[Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)、以及 [`ted_xiao_embodied_three_eras_primary_refs.md`](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md) 中的论文指针。

# 机器人学习「三个时代」：叙事透镜与一手文献

## 使用方式（重要）

中文媒体编译稿与口述访谈适合作为**动机与术语地图**，其中涉及的数字、内部代号与因果陈述可能无法逐条核验。本页把编译稿里的技术名词压成三条**阅读轴线**，每条轴线只承接可在 arXiv / 机构博客上复核的实体；完整书目见 [`sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md`](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)。

---

## 轴线 I — 存在性证明：连续控制里的端到端学习能否落地

- **游戏侧先例**：[深度强化学习游戏里程碑](../concepts/deep-rl-game-milestones.md)（DQN / AlphaGo）说明离散动作与高维观测下的端到端可行性，但与连续关节控制不可混为一谈。
- **真实机械臂规模化 RL**：[QT-Opt](../methods/qt-opt.md) 把异策 Q 学习与大规模真实抓取数据结合；视觉域对齐常讨论 [CycleGAN Sim2Real](../methods/cyclegan-sim2real.md)。
- **并行探索**：[BC-Z](../methods/bc-z.md)、[MT-Opt](../methods/mt-opt.md)、[Learning from Play](../methods/learning-from-play-lmp.md) 覆盖语言条件模仿、多任务 RL 与非结构化玩耍数据。
- **事后重标记谱系**：[HER](../methods/her.md) 与语言侧的 [DIAL](../methods/dial-instruction-augmentation.md)（VLM 指令增强）形成跨模态对照。

**知识库延伸**：[模仿学习](../methods/imitation-learning.md)、[强化学习](../methods/reinforcement-learning.md)、[Sim2Real](../concepts/sim2real.md)。

---

## 轴线 II — 基础模型：把外部多模态智能接入机器人策略

- **规划 + affordance**：[SayCan](../methods/saycan.md) 用语言模型生成候选子任务并用价值估计约束可行性。
- **策略侧 Transformer / VLA**：[Robotics Transformer（RT-1 / RT-2）](../methods/robotics-transformer-rt-series.md) 与总览页 [VLA](../methods/vla.md)。
- **跨本体数据轴**：[Open X-Embodiment](../concepts/open-x-embodiment.md) 连接 [Foundation Policy](../concepts/foundation-policy.md) 与 [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)。

**知识库延伸**：[Foundation Policy](../concepts/foundation-policy.md)、[VLA](../methods/vla.md)。

---

## 轴线 III — Scaling：评测、数据形态与产业侧闭链叙事

- **分布式真实评测**：[RoboArena](../methods/roboarena.md)。
- **闭源模型族**：[Gemini Robotics](../entities/gemini-robotics.md)（以官方博客与技术报告为准）。
- **开源 generalist**：[Octo](../methods/octo-model.md)；**商业数据叙事**：[Generalist AI](../entities/generalist-ai-robotics.md)（以公司博客为准，区分论文与市场宣传）。
- **世界模型纵览**：仓库内 [Generative World Models](../methods/generative-world-models.md)、[rl_foundation_models](../../sources/papers/rl_foundation_models.md) 索引。

**知识库延伸**：[Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)、[Data Flywheel](../concepts/data-flywheel.md)。

---

## 口述叙事 vs 可核验结论

编译稿中的「Code Yellowish」「一年半不发论文」「八万七千条轨迹」等属于**组织史叙事**：可与 RT-1 / DIAL 论文中的数据管线对照阅读，但不应单独作为事实条目写进概念定义。若需引用规模数字，优先引用对应论文或官方技术报告中的表格。

---

## 关联页面

- [Foundation Policy](../concepts/foundation-policy.md)
- [深度强化学习游戏里程碑](../concepts/deep-rl-game-milestones.md)
- [Open X-Embodiment](../concepts/open-x-embodiment.md)
- [VLA](../methods/vla.md)
- [QT-Opt](../methods/qt-opt.md) · [MT-Opt](../methods/mt-opt.md) · [BC-Z](../methods/bc-z.md)
- [Learning from Play](../methods/learning-from-play-lmp.md) · [HER](../methods/her.md) · [CycleGAN Sim2Real](../methods/cyclegan-sim2real.md)
- [SayCan](../methods/saycan.md) · [Robotics Transformer](../methods/robotics-transformer-rt-series.md) · [DIAL](../methods/dial-instruction-augmentation.md)
- [Octo](../methods/octo-model.md) · [RoboArena](../methods/roboarena.md)
- [Gemini Robotics](../entities/gemini-robotics.md) · [Generalist AI](../entities/generalist-ai-robotics.md)
- [Imitation Learning](../methods/imitation-learning.md) · [Reinforcement Learning](../methods/reinforcement-learning.md) · [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)

## 参考来源

- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
- [rl_foundation_models.md](../../sources/papers/rl_foundation_models.md)

## 推荐继续阅读

- RoboPapers 访谈原视频：https://www.youtube.com/watch?v=etPqBphTgmE
- 触发编译的微信公众号文章（二手综述）：https://mp.weixin.qq.com/s/YJYy7dRGUbykxng2gEt9gw
