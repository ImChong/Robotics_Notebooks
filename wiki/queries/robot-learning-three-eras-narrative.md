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
  - ../methods/vla.md
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

- **游戏里的端到端成功后**：真实机器人面对的是连续动作与高维观测；QT-Opt 一类工作把可扩展的异策价值学习与真实机械臂农场数据结合，证明像素输入下的抓取与闭环改进可以规模化（Kalashnikov et al., arXiv:1806.10293）。
- **并行探索**：BC-Z（语言条件模仿，arXiv:2202.02005）、MT-Opt（多任务 RL，arXiv:2104.08212）、Learning from Play（从非任务化交互中学习结构，arXiv:1903.01973）展示同一时期团队在「任务定义—数据形态—算法」上的分叉尝试。
- **与后世语言条件模型的接口**：事后重标记思想在 HER（arXiv:1707.01495）中系统化；后在语言指令场景由 DIAL（arXiv:2211.11736，一作 Ted Xiao）用 VLM 作轨迹级指令增强，与 RT-1 训练管线衔接。

**知识库延伸**：[模仿学习](../methods/imitation-learning.md)、[强化学习](../methods/reinforcement-learning.md)、[Sim2Real](../concepts/sim2real.md)。

---

## 轴线 II — 基础模型：把外部多模态智能接入机器人策略

- **规划与落地约束**：SayCan（arXiv:2204.01691）用语言模型生成候选子任务，并用习得的价值估计筛掉当前环境下不可行的步骤。
- **策略本体 Transformer 化**：RT-1（arXiv:2212.06817）把图像与指令编码为 token，并离散化动作输出；RT-2（arXiv:2307.15818）把大规模 VLM 当作骨干，形成常说的 VLA 范式。
- **跨本体数据**：Open X-Embodiment（arXiv:2310.08864）把异构机器人数据纳入统一训练叙事，与 [Foundation Policy](../concepts/foundation-policy.md)、[Embodied Scaling Laws](../concepts/embodied-scaling-laws.md) 直接相连。

**知识库延伸**：[VLA](../methods/vla.md)、[foundation policy](../concepts/foundation-policy.md)。

---

## 轴线 III — Scaling：评测、数据形态与产业侧闭链叙事

- **评测**：RoboArena（arXiv:2506.18123）代表「分布式真实世界、对比式排名」一类评估潮流，用于约束日益宽泛的 generalist policy 声称。
- **机构模型迭代**：Gemini Robotics / ER / 1.5 系列应以 Google DeepMind 博客与技术报告 PDF 为准（链接见一手索引），不宜把新闻措辞直接写成可引用事实而不附出处。
- **开源 / 商业 generalist**：Octo（arXiv:2405.12213）、π₀（arXiv:2410.24164）与各类公司博客中的数据规模声明，引用时需区分**同行评审论文**与**市场传播材料**。

**知识库延伸**：[Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)、[Data Flywheel](../concepts/data-flywheel.md)。

---

## 口述叙事 vs 可核验结论

编译稿中的「Code Yellowish」「一年半不发论文」「八万七千条轨迹」等属于**组织史叙事**：可与 RT-1 / DIAL 论文中的数据管线对照阅读，但不应单独作为事实条目写进概念定义。若需引用规模数字，优先引用对应论文或官方技术报告中的表格。

---

## 关联页面

- [Foundation Policy](../concepts/foundation-policy.md)
- [VLA](../methods/vla.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)

## 参考来源

- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
- [rl_foundation_models.md](../../sources/papers/rl_foundation_models.md)

## 推荐继续阅读

- RoboPapers 访谈原视频：https://www.youtube.com/watch?v=etPqBphTgmE
- 触发编译的微信公众号文章（二手综述）：https://mp.weixin.qq.com/s/YJYy7dRGUbykxng2gEt9gw
