---
title: "万字长文 ｜ GEN-1.5 火了，但机器人的\"上下文学习\"到底在学什么？"
author: 具身智能之心
date: "2026-08-25 17:08:41"
source: "https://mp.weixin.qq.com/s/V_Dm8kHvB2YxtGY7qScjXA"
---

# 万字长文 ｜ GEN-1.5 火了，但机器人的"上下文学习"到底在学什么？

最近，Generalist AI 发布的机器人大模型 **GEN-1.5** 展示了一个值得关注的结果：给机器人看一段几秒到十几秒的示范——可以是人手现场演示，也可以是另一台机器人此前的执行记录——它在不做任何额外训练、不更新任何参数的情况下，就能去执行一个此前从未见过的新任务。

官方公布的演示视频里，机器人在看完一段陌生任务的示范后当场上手，成功率不算完美，但确实体现出"看一次就能执行"的能力。

GEN-1.5 把这段示范称为 **physical prompt（物理提示）**：一段感觉运动序列（传感器数据加动作轨迹），被直接放进模型的输入窗口，和机器人当前正在看到的画面拼在一起。模型读完这段提示后，直接执行，中间没有梯度更新。

这类能力并不是第一次出现。五年前，GPT-3 已经在语言上展示过类似的效果：给模型几个"英文译法文"的例子，不需要重新训练，它就能把第四个单词也翻译对。这种能力被称为 **In-Context Learning（上下文学习，简称 ICL）**——模型没有被专门训练成某个任务的执行器，权重没有变化，但它从输入里给出的几个例子中"读出"了任务规律，并把这个规律应用到了新的输入上。

把这个概念从语言搬到机器人控制上并不是简单的类比替换。语言模型处理的是离散的文本 token，机器人处理的是连续的图像、力和动作，而且今年围绕"上下文"做文章的机器人工作也不止 GEN-1.5 一个：Physical Intelligence 的 π0.7 把语言指令、目标图像、episode metadata 和历史轨迹一起塞进同一个上下文；RoboTTT 把机器人策略能处理的视觉运动历史推到了 8000 步。"上下文"在机器人领域重新变成了一个高频词。

但把这些工作统一称为"机器人 ICL"会掩盖一个问题：**它们往上下文里装的东西并不是一类。**有的装任务示范，有的装人的运动意图，有的装策略的执行风格，还有的装机器人自己的探索轨迹。共用一个名词，并不意味着它们在解决同一个问题。这篇文章尝试把这个笼统的说法拆开，看清楚不同工作各自往上下文里放了什么，又分别解决了什么。

（正文其余章节：马尔可夫假设与上下文必要性；三类不确定性（映射选择 / 状态估计 / 映射本身）；遥操作示范、人类视频、任务无关随机运动、预训练规模涌现、π0.7 选择与记忆、TTT 正交路径、开放问题与速查表。完整归纳见 `sources/blogs/wechat_embodied_heart_robot_icl_gen15_survey_2026-08-25.md`。）

## 参考文献（文内编号）

1. One-Shot Imitation Learning，NeurIPS 2017
2. One-Shot Imitation from Observing Humans via Domain-Adaptive Meta-Learning，arXiv 2018
3. ICRT: In-Context Imitation Learning via Next-Token Prediction，ICRA 2025
4. Instant Policy: In-Context Imitation Learning via Graph Diffusion，ICLR 2025
5. Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics，RSS 2024
6. Action Tokenizer Matters in In-Context Imitation Learning，IROS 2025
7. Behavior Prompting Policy: Demonstrations as Prompts for Manipulation，arXiv 2026
8. StellaVLA: In-Context Structured Demonstration for Generalizable Vision-Language-Action Models，arXiv 2026
9. SynthICL: Scalable In-context Imitation Learning with Synthetic Data，arXiv 2026
10. RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models，CoRL 2025
11. Vid2Robot: End-to-end Video-conditioned Policy Learning with Cross-Attention Transformers，RSS 2024
12. MimicDroid: In-Context Learning for Humanoid Robot Manipulation from Human Play Videos，ICRA 2026
13. Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation，arXiv 2025
14. In-Context World Modeling for Robotic Control，arXiv 2026
15. Qwen-RobotManip Technical Report，arXiv 2026
16. GEN-1.5: Embodied Foundation Models are One-Shot Learners，Generalist AI 2026
17. π0.7: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities，Physical Intelligence 2026
18. MemoryVLA，ICLR 2026
19. MemER，ICLR 2026
20. ContextVLA，arXiv 2025
21. MEM: Multi-Scale Embodied Memory for Vision Language Action Models，arXiv 2026
22. HiMe，ICML 2026
23. BPP: Long-Context Robot Imitation Learning by Focusing on Key History Frames，arXiv 2026
24. Gated Memory Policy: In-Context Memorization and Adaptation，arXiv 2026
25. RoboTTT: Context Scaling for Robot Policies，arXiv 2026
26. VANE: Reliable Test-Time Training for Vision-Language-Action Models via Future Visual Representation Prediction，arXiv 2026
