---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, reinforcement-learning, rhythmic-contact, bimanual, polimi, unitree]
status: complete
updated: 2026-09-28
arxiv: "2507.11498"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../methods/ppo.md
  - ../concepts/humanoid-policy-reward-functions.md
  - ../tasks/bimanual-manipulation.md
  - ../methods/table-tennis-strategy-skill-learning.md
  - ./paper-coordinated-badminton-skills-anymal.md
sources:
  - ../../sources/papers/humanoid_pnb_robot-drummer.md
summary: "人形在灵巧、平衡、行走上进步显著，但在音乐表演等表现性领域的角色仍少被探索。本文提出 Robot Drummer，通过一连串定时接触完成打鼓，把问题表述成节奏接触链（Rhythmic Contact Chain）。系统把乐曲分解成定长片段，并行用强化学习训练。在 30+ 首摇滚、金属、爵士曲目上测试，取得高 F1 分数，并涌现出交叉臂击打（cross-arm strikes）与自适应鼓棒分配（adaptive stick assignments）等行为，能完成数分钟级的多肢协调演奏。"
---

# Robot Drummer

**Robot Drummer: Learning Rhythmic Skills for Humanoid Drumming** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

人形在灵巧、平衡、行走上进步显著，但在音乐表演等表现性领域的角色仍少被探索。本文提出 Robot Drummer，通过一连串定时接触完成打鼓，把问题表述成节奏接触链（Rhythmic Contact Chain）。系统把乐曲分解成定长片段，并行用强化学习训练。在 30+ 首摇滚、金属、爵士曲目上测试，取得高 F1 分数，并涌现出交叉臂击打（cross-arm strikes）与自适应鼓棒分配（adaptive stick assignments）等行为，能完成数分钟级的多肢协调演奏。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Rhythmic Contact Chain | 节奏接触链，一连串定时接触 |
| Timed Contact | 定时接触，在准确时刻击鼓 |
| F1 Score | 衡量击打准确性的指标 |
| Cross-Arm Strike | 交叉臂击打（涌现策略） |
| Stick Assignment | 鼓棒分配（哪只手打哪面鼓） |
| Parallel RL | 并行强化学习 |

## 为什么重要

- **把表现性任务转成"定时接触序列"**是可学化的关键抽象；
- **分段并行 RL**对长时程节奏任务有效；
- **表现性领域（音乐）**是高动态多肢协调的新颖试金石，呼应羽毛球/足球等体育任务；
- 涌现的拟人策略显示 RL 能发现高效协调方式。

## 解决什么问题

人形**音乐表演（打鼓）**少被探索，难点： - 打鼓是**精确定时的多肢接触**序列； - 乐曲**长时程**，直接 RL 难； - 要**多肢协调**（双臂 + 鼓棒分配）。

Robot Drummer 要：把打鼓建模成可学的**定时接触序列**，让人形演奏真实曲目。

## 核心机制

1. **节奏接触链表述**：把打鼓转成定时接触序列；
2. **分段并行 RL**：破解长时程乐曲；
3. **涌现拟人策略**：交叉臂击打、自适应鼓棒分配；
4. **真实曲目验证**：30+ 摇滚/金属/爵士高 F1。

方法拆解（深读笔记小节）：节奏接触链（Rhythmic Contact Chain）；分段并行 RL；涌现拟人策略；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming.html> |
| arXiv | <https://arxiv.org/abs/2507.11498> |
| 源码 | **未开源**：项目页 <https://robotdrummer.github.io> 标注 “Code (Coming Soon)”，截至 2026-09-28 未列仓库 |
| 作者 | Asad Ali Shahid、Francesco Braghin、Loris Roveda（米兰理工 / IDSIA） |
| 发表 | 2025 年 7 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**设置**：Isaac Gym 中的 Unitree G1，只控制上身 15 DoF（双臂各 7 + 躯干 1），鼓棒刚性固定在手上。乐曲转为「节奏接触链」（RCC），评测用每个时间步「期望击打 vs 实际击打」的精确率、召回率与 F1。共训练 30 多个专才策略（每首歌一个）。

| 曲目（无复音，≤2 个同时击打） | 鼓数 | nPVI（节奏不规则度） | BPM | F1 |
|------|---:|---:|---:|---:|
| David Bowie – Rebel Rebel | 2 | 8.76 | 130 | 0.985 |
| The White Stripes – Seven Nation Army | 4 | 6.71 | 120 | 0.977 |
| Nirvana – Lithium | 6 | 22.10 | 112 | 0.954 |
| Jimi Hendrix – Fire | 6 | 29.48 | 156 | 0.943 |
| Dave Brubeck – Take Five | 4 | 51.86 | 167 | 0.908 |
| Linkin Park – In The End | 6 | 32.68 | 86 | 0.901 |
| Bon Jovi – Livin' on a Prayer | 6 | 82.93 | 123 | 0.885 |
| The Police – Roxanne | 6 | 21.19 | 136 | 0.878 |

- **难度来源**：节奏不规则度 nPVI 与 F1 的 Spearman 相关最强（ρ = −0.52），鼓数其次（ρ = −0.42）。Roxanne 在不同鼓之间快速切换，策略倾向于停在高频目标上而漏掉其余击打。
- **消融**（5 次运行）：完整密集接触奖励（正确 + 错误 + 漏击）明显优于只奖励正确击打；观测中去掉接触目标，F1 接近 0；用连续相位变量代替接触目标同样表现差。
- **时间分段**：最终 F1 相近，但整首训练约 8–9 小时收敛，分段训练只需 2–3 小时。
- **多曲目单策略**：6 首歌合训在每首上都明显低于专才策略，8 首合训更差（负迁移）。
- **听众研究**（15 人，5 分制）：节奏一致性 3.6、表现力 3.4、自然度 2.5、整体喜爱度 3.1。

## 与其他工作对比

| 做法 | 时序目标的表示 | 与 Robot Drummer 的差异 |
|------|------|------|
| 相位条件策略（论文消融） | 连续归一化相位 | 只知道「进行到哪」，不知道「下一下打哪」，性能很差 |
| 动作模仿类（跟踪人类演奏动作） | 参考动作轨迹 | 需要演奏动作数据；本文只用鼓谱生成接触目标 |
| [乒乓球技能学习](../methods/table-tennis-strategy-skill-learning.md) / [羽毛球](./paper-coordinated-badminton-skills-anymal.md) | 由球的轨迹决定击打时刻 | 同为定时接触任务；鼓谱时刻已知，但接触序列更长、更密集 |

## 结论

**Robot Drummer 的关键一步在表述层而非算法层：把「打鼓」重写成节奏接触链（一串定时接触），长时程音乐表演才第一次变成可标准化训练的 RL 问题。**

- 真正可学化的抽象是「定时接触」：接触链把演奏转成时刻明确的接触目标，评价也随之落到 F1 这类击打准确性指标上。
- 长时程靠工程手段化解：乐曲被切成定长片段并行训练，这是能完成数分钟级多肢协调演奏的直接原因。
- 涌现行为是额外收获而非设计目标：交叉臂击打与自适应鼓棒分配未被显式指定，说明双臂之间的分工可以交给 RL 自行发现。
- 证据覆盖 30+ 首摇滚/金属/爵士曲目并取得高 F1；本库把它归入 06_Manipulation，视角是高动态多肢协调的试金石，与羽毛球、足球等体育任务同类。
- 边界也很清楚：全部是仿真结果，专才策略 F1 在 0.88–0.99，多曲目单策略出现负迁移；节奏不规则度（nPVI）是最主要的难度来源。

## 局限与风险

- **只在仿真中验证**：没有建模声音、鼓棒回弹与握持柔顺性，策略只优化击打时机与位置，不涉及音色和力度。
- **迁移真机的前提**：作者认为需要柔顺的手–棒接口，以及补偿未建模动力学的残差控制；可借助输出 MIDI 的电子鼓对齐时机与力度。
- **专才策略**：每首歌单独训练，多曲目合训出现负迁移。
- **不规则节奏仍难**：nPVI 高、鼓间快速切换的曲目 F1 降到 0.88 左右。
- **开源边界**：代码标注 Coming Soon；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 强化学习训练：[ppo](../methods/ppo.md)
- 密集接触奖励设计：[humanoid-policy-reward-functions](../concepts/humanoid-policy-reward-functions.md)
- 双臂协调：[bimanual-manipulation](../tasks/bimanual-manipulation.md)
- 高动态定时击打类技能对照：[table-tennis-strategy-skill-learning](../methods/table-tennis-strategy-skill-learning.md)
- 体育类高动态多肢协调对照：[paper-coordinated-badminton-skills-anymal](./paper-coordinated-badminton-skills-anymal.md)

## 参考来源

- [humanoid_pnb_robot-drummer.md](../../sources/papers/humanoid_pnb_robot-drummer.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming.html>
- 论文：<https://arxiv.org/abs/2507.11498>
- 论文正文（Table I–II、多曲目实验与局限）：<https://arxiv.org/html/2507.11498>

## 推荐继续阅读

- [机器人论文阅读笔记：Robot Drummer](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming/Robot_Drummer__Learning_Rhythmic_Skills_for_Humanoid_Drumming.html)
- 项目页：<https://robotdrummer.github.io>
