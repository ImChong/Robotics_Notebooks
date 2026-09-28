---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, in-context-learning, human-video, ut-austin, fourier]
status: complete
updated: 2026-09-28
arxiv: "2509.09769"
venue: "ICRA 2026"
code: https://github.com/UT-Austin-RPL/mimicdroid-robocasa/tree/latest
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../concepts/robot-in-context-learning.md
  - ../methods/wilor.md
  - ./paper-notebook-robocasa-large-scale-simulation-of-everyday-task.md
  - ./paper-notebook-okami-teaching-humanoid-robots-manipulation-skil.md
  - ./paper-ego-03-egomimic.md
sources:
  - ../../sources/papers/humanoid_pnb_mimicdroid.md
summary: "目标是让人形从少量视频示例高效解决新操作任务。上下文学习（ICL）因测试时数据高效、快速适应而有前景，但现有 ICL 方法依赖费力的遥操作数据，难规模化。本文用人类玩耍视频（human play videos）——人们自由与环境交互的连续、无标注视频——作为可扩展、多样的训练源。提出 MimicDroid：仅用人类玩耍视频做训练，抽取行为相似的轨迹对，训练策略以一条轨迹为条件预测另一条的动作，从而获得测试时适应新物体/环境的 ICL 能力。为弥合具身差距，先用运动学相似性把 RGB 视频估计的人手腕姿态重定向到人形；训练时随机块遮挡（patch masking）降低对人类特有线索的过拟合、增强对视觉差异的鲁棒。作者还提出一个开源仿真基准（难度递增）评估少样本学习；MimicDroid 优于 SOTA，真机成功率近两倍。"
---

# MimicDroid

**MimicDroid: In-Context Learning for Humanoid Robot Manipulation from Human Play Videos** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

目标是让人形从少量视频示例高效解决新操作任务。上下文学习（ICL）因测试时数据高效、快速适应而有前景，但现有 ICL 方法依赖费力的遥操作数据，难规模化。本文用人类玩耍视频（human play videos）——人们自由与环境交互的连续、无标注视频——作为可扩展、多样的训练源。提出 MimicDroid：仅用人类玩耍视频做训练，抽取行为相似的轨迹对，训练策略以一条轨迹为条件预测另一条的动作，从而获得测试时适应新物体/环境的 ICL 能力。为弥合具身差距，先用运动学相似性把 RGB 视频估计的人手腕姿态重定向到人形；训练时随机块遮挡（patch masking）降低对人类特有线索的过拟合、增强对视觉差异的鲁棒。作者还提出一个开源仿真基准（难度递增）评估少样本学习；MimicDroid 优于 SOTA，真机成功率近两倍。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| ICL | In-Context Learning，上下文学习 |
| Play Video | 玩耍视频，自由交互的无标注视频 |
| Trajectory Pair | 轨迹对，行为相似的两段 |
| Retargeting | 重定向（人手腕→人形） |
| Patch Masking | 随机块遮挡，防过拟合 |
| Few-shot | 少样本 |

## 为什么重要

- **人类玩耍视频是海量免费的 ICL 训练源**，比遥操作更可扩展；
- **ICL 让人形"看几个例子就会"**，是快速适应的诱人范式；
- **随机块遮挡**是缓解人-机视觉差异过拟合的简单有效技巧；
- 与 In-N-On、Masquerade 等"从人类视频学操作"路线互补。

## 解决什么问题

让人形**少样本快速学新任务**： - ICL 有前景，但**依赖遥操作数据**，难规模化； - 想用**人类玩耍视频**（海量、无标注），但有**具身差距**与**人类特有线索过拟合**。

MimicDroid 要：**仅用人类玩耍视频**训练出有 ICL 能力的人形操作策略。

## 核心机制

1. **仅用人类玩耍视频的 ICL**：摆脱对遥操作数据的依赖；
2. **轨迹对条件预测**：获得测试时少样本适应能力；
3. **重定向 + 块遮挡**：弥合具身差距、防过拟合；
4. **开源基准 + 真机≈2×SOTA**。

方法拆解（深读笔记小节）：从玩耍视频抽轨迹对、学 ICL；重定向人手腕姿态；随机块遮挡防过拟合；基准与结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos.html> |
| arXiv | <https://arxiv.org/abs/2509.09769> |
| 源码 | **部分开源**：[mimicdroid-robocasa（latest 分支）](https://github.com/UT-Austin-RPL/mimicdroid-robocasa/tree/latest) 提供基于 RoboCasa 的 L1–L3 基准环境与轨迹回放；[数据集](https://huggingface.co/datasets/Rutav/MimicDroidDataset) 在 Hugging Face。截至 2026-09-28 未见 ICL 策略训练代码 |
| 作者 | Rutav Shah、Shuijing Liu、Zhenyu Jiang、Mingyo Seo、Roberto Martín-Martín、Yuke Zhu（UT Austin） |
| 发表 | 2025 年 9 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**设置**：在 RoboCasa 上自建少样本基准，分三级泛化——L1 见过物体 + 见过环境、L2 新物体、L3 新物体 + 新厨房；每级 4 个任务。训练数据为「玩耍视频」：仿真 8 小时（约 320k 步，自由浮动双手 Abstract 具身采集），真机 45 分钟（约 80k 帧，人手直接操作）。评测具身为 Abstract 手与 Fourier GR1。

仿真成功率（论文 Table I）：

| 方法 | Abstract L1 / L2 / L3 | GR1 L1 / L2 / L3 |
|------|------|------|
| H2R（目标图像条件） | 0.03 / 0.05 / 0.03 | 0.03 / 0.00 / 0.00 |
| Vid2Robot（视频条件） | 0.44 / 0.40 / 0.12 | 0.41 / 0.23 / 0.11 |
| PEFT（测试时微调） | 0.47 / 0.35 / 0.00 | 0.29 / 0.21 / 0.01 |
| MimicDroid 去掉视觉遮挡 | 0.59 / **0.51** / 0.22 | 0.37 / 0.35 / 0.09 |
| **MimicDroid** | **0.73** / 0.39 / **0.27** | **0.59** / **0.44** / **0.26** |

- **ICL vs 条件化基线**：相对任务条件化方法，Abstract / GR1 分别 +14% / +18%；相对 PEFT 分别 +29% / +26%。PEFT 在 L3 基本失效（论文归因于分布偏移下的遗忘）。
- **真机 GR1**（只用人类玩耍视频训练）：L1 / L2 / L3 = **0.53 / 0.23 / 0.08**，Vid2Robot 为 0.28 / 0.08 / 0.00，约两倍。
- **视觉遮挡消融**：去掉随机块遮挡，Abstract → GR1 迁移平均掉 17%（完整方法只掉 3%）；真机上随机块遮挡（0.53）与 EgoMimic 的手部涂黑方案（0.58）相当，但不需要 SAM 分割。
- **上下文数量**：1→3 个示例持续提升，4–6 个反而下降（受训练时上下文长度限制）。

## 与其他工作对比

| 工作 | 适应方式 | 与 MimicDroid 的差异 |
|------|------|------|
| Vid2Robot / H2R | 以人类视频或目标图为任务条件 | 上下文里没有「观测–动作」对，无法做 ICL，真机约为 MimicDroid 的一半 |
| PEFT 测试时微调 | 梯度更新 | 需要训练时间，L3 大分布偏移下遗忘严重 |
| [OKAMI](./paper-notebook-okami-teaching-humanoid-robots-manipulation-skil.md) | 单视频 + 物体感知重定向 | 同组前作，逐任务生成计划；MimicDroid 改为一次训练、测试时看几个例子即可 |
| [EgoMimic](./paper-ego-03-egomimic.md) | 人 / 机器人数据联合训练 | 用手部涂黑 + 红线弥合视觉差距；MimicDroid 用随机块遮挡达到接近效果 |

## 结论

**MimicDroid 把上下文学习的数据瓶颈从遥操作换成人类玩耍视频，代价是技术负担整体转移到跨具身对齐与防过拟合上。**

- 机制核心是「轨迹对条件预测」：从连续无标注的玩耍视频中抽取行为相似的轨迹对，训练策略以其中一条为条件预测另一条的动作，测试时才具备对新物体/新环境的少样本适应能力。
- 换来可扩展性的同时必须补两处：先按运动学相似性把 RGB 视频估计的人手腕姿态重定向到人形以弥合具身差距，再用随机块遮挡压住对人类特有线索的过拟合——这两项设计正对应本方法最集中的风险点。
- 报告的收益是优于 SOTA、真机成功率近两倍；同时配套一个难度递增的开源仿真基准专门评估少样本学习，评测口径与训练范式是配套设计的。
- 适用边界在「看几个例子就会」的快速适应，而非从零学全新技能：真机 L1 成功率 0.53，到 L3（新物体 + 新环境）只剩 0.08。
- 与 In-N-On、Masquerade 等「从人类视频学操作」路线互补，本页强调的差异是测试时适应（ICL）而非训练时模仿。

## 局限与风险

- **依赖高质量玩耍视频**：目前是专门录制的厨房玩耍视频，尚未扩展到 YouTube 级野外视频（论文自述）。
- **手部估计失效场景**：动作来自现成手部姿态估计（WiLoR），手伸进柜子或被家具挡住时无法提取动作。
- **只学「怎么做」不学「为什么」**：演示被当作低层状态–动作序列，无法在语义等价但动作不同的任务间泛化。
- **上下文长度瓶颈**：超过 3 个示例后性能下降，受训练时 transformer 上下文长度限制。
- **L3 绝对成功率仍低**：GR1 仿真 L3 仅 0.26，真机 L3 仅 0.08，离可用还有距离。
- **开源边界**：只放出基准环境与数据集，ICL 策略训练代码未见；源码运行时序图 **不适用**（无可运行的策略训练 / 推理入口）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 上下文学习概念：[robot-in-context-learning](../concepts/robot-in-context-learning.md)
- 从视频抽手部动作所用的手部重建：[wilor](../methods/wilor.md)
- 基准底座 RoboCasa：[paper-notebook-robocasa-large-scale-simulation-of-everyday-task](./paper-notebook-robocasa-large-scale-simulation-of-everyday-task.md)
- 同组单视频重定向路线：[paper-notebook-okami-teaching-humanoid-robots-manipulation-skil](./paper-notebook-okami-teaching-humanoid-robots-manipulation-skil.md)
- 手部遮挡策略对照：[paper-ego-03-egomimic](./paper-ego-03-egomimic.md)

## 参考来源

- [humanoid_pnb_mimicdroid.md](../../sources/papers/humanoid_pnb_mimicdroid.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos.html>
- 论文：<https://arxiv.org/abs/2509.09769>
- 论文正文（Table I、真机结果与局限节）：<https://arxiv.org/html/2509.09769>
- 基准代码：<https://github.com/UT-Austin-RPL/mimicdroid-robocasa/tree/latest>

## 推荐继续阅读

- [机器人论文阅读笔记：MimicDroid](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos.html)
- 项目页：<https://ut-austin-rpl.github.io/MimicDroid/>
- 数据集：<https://huggingface.co/datasets/Rutav/MimicDroidDataset>
