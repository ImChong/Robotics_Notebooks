# Rich Sutton — The Bitter Lesson

- **类型**：blog / essay（Incomplete Ideas）
- **作者**：Richard S. Sutton
- **原始链接**：<http://incompleteideas.net/IncIdeas/BitterLesson.html>
- **发布日期**：2019-03-13
- **收录日期**：2026-07-14
- **站点索引**：[incompleteideas-net-rich-sutton.md](../sites/incompleteideas-net-rich-sutton.md)

## 一句话

70 年 AI 研究的最大教训：**能随算力规模扩展的通用方法（search 与 learning）最终最有效**；把研究者的人类领域知识硬编码进系统虽短期有效，长期往往 plateau 并阻碍进一步突破。

## 为什么值得保留

- **Scaling 叙事的 RL 学派源头之一**：早于具身 scaling laws 与 LLM scaling，明确把 **Moore 定律 / 算力成本下降** 作为 AI 进步主因。
- **对机器人学习的间接约束**：Sim2Real 领域知识、手工 reward shaping、专用状态特征等「人类知识捷径」需与 **可扩展的 search/learning** 权衡——与本站 [Embodied Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md)、[Model-Based RL](../../wiki/methods/model-based-rl.md) 讨论同频。
- **历史案例密集**：国际象棋（Deep Blue）、围棋（AlphaGo）、语音识别（HMM→统计→深度学习）、计算机视觉（SIFT/边缘→卷积）均呈现同一模式。

## 核心论点（原文归纳）

### 历史模式（四点）

1. AI 研究者常试图把知识 built into agents
2. 这在短期几乎总有帮助，且令研究者个人满足
3. 长期会 plateau，甚至抑制进一步进展
4. 突破最终来自 **对立路线**：通过 scaling computation 的 search 与 learning

### 两类可任意扩展的通用方法

- **Search**（如博弈树搜索、规划）
- **Learning**（如 self-play 学 value function、大规模训练）

二者共同点是能把 **海量算力** 用到问题上。

### 关于「心智内容」

心智的实际内容极其复杂；不应试图用简单先验（空间、物体、多智能体、对称性等）built in——其复杂度 endless。应 built in 的是 **能发现并捕获任意复杂性的 meta-methods**（能找好近似，但搜索应交给方法而非研究者）。

> We want AI agents that can discover like we can, not which contain what we have discovered.

## 对 wiki 的映射

- [wiki/concepts/bitter-lesson.md](../../wiki/concepts/bitter-lesson.md)
- [wiki/entities/richard-sutton.md](../../wiki/entities/richard-sutton.md)
- 交叉：[wiki/concepts/embodied-scaling-laws.md](../../wiki/concepts/embodied-scaling-laws.md)

## 参考链接

- 原文：<http://incompleteideas.net/IncIdeas/BitterLesson.html>
- 作者主页：<http://incompleteideas.net/>
