# Richard Sutton — incompleteideas.net 一手资料索引

- **类型**：个人站点 / 论文、教材、课程、博客与 RL 一手资料总索引（原始资料归档）
- **收录日期**：2026-07-14
- **主链接**：<http://incompleteideas.net/>
- **备用主页**：<http://richsutton.com/>
- **抓取说明**：以 **2026-07-14** 对 `incompleteideas.net` 首页及子页公开 HTML 为准；站点为经典静态 HTML，部分子资源托管于 UAlberta `rlai.cs.ualberta.ca` 与 Google Docs。

## 一句话

**Richard S. Sutton** 个人学术主页：强化学习（RL）奠基人之一、Turing Award（2019，与 Andrew Barto 共享）得主；集中索引 **Sutton & Barto 教材**、Alberta RL MOOC、RL FAQ、Incomplete Ideas 博客、研究提案与经典工具（tile coding 等）——是理解 **MDP/TD/选项/GVF** 与 Sutton 研究哲学的**一手入口**。

## 为什么值得保留

- **RL 标准教材与课程源头**：*Reinforcement Learning: An Introduction*（2nd ed., 2018）官方页、Coursera/UAlberta RL 专项、RL FAQ 均从此站链出；本库 [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[MDP](../../wiki/formalizations/mdp.md)、[Bellman Equation](../../wiki/formalizations/bellman-equation.md) 等页的**权威外部参照**应指向此处而非镜像 PDF。
- **方法论一手论述**：*The Bitter Lesson*（2019）、*The One-Step Trap*（2024）等 Incomplete Ideas 博文直接影响 **scaling / search / learning** 与 **model-based RL** 争论；与 [Embodied Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md)、[Model-Based RL](../../wiki/methods/model-based-rl.md) 交叉。
- **Alberta RL 学派脉络**：Alberta Plan、RLAI slogans、Horde/GVF/Options 等长期研究线在此有提案与演讲索引，便于与本站 streaming RL、foundation policy 等当代路线对照。

## 人物与机构（2026-07-14 首页）

| 维度 | 内容 |
|------|------|
| 姓名 | Richard S. Sutton |
| 机构 | Oak Lab 创始人；UAlberta Computing Science 教授；Amii CIFAR AI Chair；Openmind Research Institute 创始人兼首席科学家 |
| 联系 | rich@richsutton.com；X: @RichardSSutton |
| 学术荣誉 | 2019 ACM Turing Award（与 Andrew Barto，「for conceptual and engineering breakthroughs that have made deep reinforcement learning a critical component of computing」） |

## 核心一手资料索引

### 教材与教学

| 资源 | 链接 | 说明 |
|------|------|------|
| **Sutton & Barto RL 教材（2nd ed.）** | <http://incompleteideas.net/book/the-book-2nd.html> | MIT Press 2018；含 PDF、errata、slides、代码解答交换 |
| 第 1 版 | <http://incompleteideas.net/book/first/the-book.html> | 1998 版归档 |
| **RL MOOC** | <https://www.ualberta.ca/admissions-programs/online-courses/reinforcement-learning/index.html> | UAlberta + Coursera；Martha & Adam White 主讲 |
| Coursera 专项 | <https://www.coursera.org/specializations/reinforcement-learning> | 四课程专项 |
| **RL FAQ** | <http://incompleteideas.net/RL-FAQ.html> | Sutton 编纂（2001–2004）；RL 定义、教学书目、函数逼近、连续动作等 FAQ |
| Turing Award 讲解视频 | <https://www.youtube.com/watch?v=RrXibq7-W6o> | ACM 官方 RL 科普 |
| CMPUT 课程页（历史） | `rlai.cs.ualberta.ca/cmput*` | 609/325/607/366 等课程材料链接 |

### 研究提案与计划（一手 PDF / Doc）

| 标题 | 链接 |
|------|------|
| The Alberta Plan for AI Research | <https://arxiv.org/pdf/2208.11173.pdf> |
| Top 10 Readings for my RL approach to AI | <https://docs.google.com/document/d/1juudZLXpqMsuAXg7zGFlkRdBf8hffDzSChWkHAmJci0/edit> |
| 2025 CCAI-chair proposal | `CCAIprop2025.pdf`（站内） |
| 2020 CCAI chair research proposal | `CCAIprop2020.pdf` |
| 2022 NSERC technical proposal | `NSERCtechnical2022.pdf` |
| 2012 iCORE research proposal | `icoreprop2012.pdf` |

### Incomplete Ideas 博客（高信号子集）

| 日期 | 标题 | 链接 | 备注 |
|------|------|------|------|
| 2019-03-13 | **The Bitter Lesson** | <http://incompleteideas.net/IncIdeas/BitterLesson.html> | 算力 + search/learning 胜过内置人类知识；见 [sources/blogs/sutton_bitter_lesson.md](../blogs/sutton_bitter_lesson.md) |
| 2024-07-18 | **The One-Step Trap** | <http://incompleteideas.net/IncIdeas/OneStepTrap.html> | 反对「单步模型 rollout 得长期预测」；主张 options/GVF 时序抽象 |
| 2016-07-09 | The Definition of Intelligence | <http://incompleteideas.net/IncIdeas/DefinitionOfIntelligence.html> | McCarthy 定义 + 目标作为 observer stance |
| 2001-11-15 | Verification, The Key to AI | <http://incompleteideas.net/IncIdeas/KeytoAI.html> | 智能体须能自验证知识 |
| 2008 | 14 principles of experience oriented intelligence | 站内 | 经验导向智能原则 |
| 2019 | Turing award acceptance speech | `IncIdeas/TuringSpeech.html` | 图灵奖获奖演讲 |
| 2019 | Podcast re My Life So Far | Eye on AI EP11 + transcript | 生涯回顾 |

完整列表见首页 **Incomplete Ideas** 小节。

### 演讲与写作

| 资源 | 链接 |
|------|------|
| Talks 索引 | <http://incompleteideas.net/Talks/Talks.html> |
| A Perspective on Intelligence | Talks 子页 |
| Advice for would-be RL researchers | Talks 子页 |
| Publications | <http://incompleteideas.net/publications.html> |
| Google Scholar | <https://scholar.google.ca/citations?user=6m4wv6gAAAAJ> |
| CV | `suttonCV.pdf` |

### RL 研究 slogans（RLAI）

<http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/richsprinciples.html> — Sutton 归纳的 10 条研究口号，摘录：

1. Approximate the solution, not the problem
2. Drive from the problem
3. Take the agent's point of view
4. Don't ask the agent to achieve what it can't measure
5. Don't ask the agent to know what it can't verify
6. Set measurable goals for subparts of the agent
7. Discriminative models are usually better than generative models
8. Work by orthogonal dimensions. Work issue by issue
9. Work on ideas, not software
10. Experience is the data of AI

### 软件与工具

| 工具 | 链接 | 说明 |
|------|------|------|
| Tile Coding v3 | <http://incompleteideas.net/tiles/tiles3.html> | 连续状态函数逼近经典实现 |
| RandomMDPs.lisp | `RandomMDPs.html` | 表格/因子化 MDP 学习规划 |
| RLAI / RLPark | <http://rlai.net>、<http://rlpark.github.com/> | Alberta RL 研究生态 |

## RL FAQ 要点摘录（教学向）

- **RL 定义**：从与环境的交互及行动后果中学习，而非显式教学；数学框架以 MDP 为主。
- **推荐入门**：Sutton & Barto 教材；更形式化见 Bertsekas & Tsitsiklis *Neuro-Dynamic Programming*；短篇见 Kaelbling et al. 1996 survey 与 Barto-Sutton-Watkins 1990。
- **大状态空间**：必须用函数逼近；Q-learning + 线性逼近已知不 sound，on-policy 线性预测有 Tsitsiklis–Van Roy 等结果。
- **连续动作**：需采样替代枚举；actor-critic 是经典路线之一。
- **Tile coding**：优于 naive 网格的关键在于 **多个偏移网格的叠加**（generalization），而非单纯细分网格。

## 对 wiki 的映射

- 升格页面：[wiki/entities/richard-sutton.md](../../wiki/entities/richard-sutton.md)
- 教材实体：[wiki/entities/sutton-barto-rl-book.md](../../wiki/entities/sutton-barto-rl-book.md)
- 概念页：[wiki/concepts/bitter-lesson.md](../../wiki/concepts/bitter-lesson.md)
- 交叉更新：[wiki/methods/reinforcement-learning.md](../../wiki/methods/reinforcement-learning.md)、[wiki/methods/model-based-rl.md](../../wiki/methods/model-based-rl.md)、[wiki/concepts/embodied-scaling-laws.md](../../wiki/concepts/embodied-scaling-laws.md)

## 推荐继续阅读（外部）

- [incompleteideas.net](http://incompleteideas.net/) — 一手总索引
- [Sutton & Barto 教材 2nd ed.](http://incompleteideas.net/book/the-book-2nd.html)
- [The Bitter Lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html)
- [Alberta RL Coursera 专项](https://www.coursera.org/specializations/reinforcement-learning)
