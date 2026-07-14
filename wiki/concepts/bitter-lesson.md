---
type: concept
tags: [scaling-laws, reinforcement-learning, methodology, ai-history]
status: complete
updated: 2026-07-14
related:
  - ../entities/richard-sutton.md
  - ../concepts/embodied-scaling-laws.md
  - ../methods/reinforcement-learning.md
  - ../methods/model-based-rl.md
  - ../concepts/data-flywheel.md
  - ../entities/paper-from-agi-to-asi.md
sources:
  - ../../sources/blogs/sutton_bitter_lesson.md
  - ../../sources/sites/incompleteideas-net-rich-sutton.md
summary: "The Bitter Lesson：AI 进步最终来自能随算力扩展的通用方法（search 与 learning），而非把人类领域知识硬编码进系统。"
---

# The Bitter Lesson（惨痛教训）

**The Bitter Lesson**：Richard Sutton 2019 年提出的 AI 方法论观察——**通用、可随算力规模扩展的方法（search 与 learning）长期压倒内置人类领域知识的路线**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AI | Artificial Intelligence | 短文讨论 70 年 AI 研究史的方法论教训 |
| RL | Reinforcement Learning | Sutton 语境中 learning 的代表领域之一 |
| CV | Computer Vision | 短文案例：SIFT/边缘特征 → 卷积深度学习 |
| ASR | Automatic Speech Recognition | 短文案例：语音学先验 → HMM → 深度学习 |

## 为什么重要

- **Scaling 话语的 RL 源头**：在 LLM / 具身 scaling laws 流行之前，已明确把 **算力成本下降** 作为 AI 突破的主因，并命名 **search** 与 **learning** 为两类可任意扩展的通用技术。
- **对机器人研究的警示**：手工特征、专用 reward 工程、强领域先验的 sim 结构虽短期有效，但若阻碍 **可扩展学习管线**（更多仿真 rollouts、更大策略网络、更多真机数据），长期可能 plateau。
- **与具身 scaling 的层级关系**：[Embodied Scaling Laws](./embodied-scaling-laws.md) 讨论 **轨迹/参数/任务** 幂律；Bitter Lesson 讨论 **宏观方法论选择**——二者互补而非替代。

## 核心机制：四步历史模式

```mermaid
flowchart LR
  A["研究者注入人类领域知识"] --> B["短期性能提升"]
  B --> C["长期 plateau / 阻碍扩展"]
  C --> D["search + learning 随算力突破"]
  D --> E["成功带苦涩：击败受青睐的人类先验路线"]
```

### 两类可扩展通用方法

| 方法类 | 含义 | 短文案例 |
|--------|------|----------|
| **Search** | 用算力探索决策空间 | 国际象棋 Deep Blue 深度搜索；围棋 MCTS |
| **Learning** | 用数据/自对弈学 value/policy | AlphaGo self-play；语音识别深度学习 |

### 关于 built-in 知识

心智内容极其复杂，**不应**把空间、物体、对称性等 endless 复杂度 built in；应 built in **能发现任意复杂性的 meta-methods**。智能体应 **像人类一样发现**，而非 **装载人类已发现的内容**。

## 工程实践：如何「不重复惨痛教训」

| 倾向 | 更可扩展的替代 |
|------|----------------|
| 手工设计状态特征 | 让策略网络从原始感知学习（配合足够数据与算力） |
| 复杂 reward shaping 规则 | 简单 reward + 大规模 RL / curriculum |
| 专用解析控制器 + 小网络补丁 | 端到端或大规模 sim rollouts（在 safety 约束内） |
| 小规模专家演示 + 复杂领域模型 | 数据飞轮 + 可扩展 IL/RL 管线 |

**注意**：Bitter Lesson 不是「领域知识永远无用」——仿真物理、安全约束、硬件极限仍是工程必需；短文强调的是 **研究资源在 human-knowledge 捷径 vs 可扩展方法之间的权衡**。

## 局限与风险

- **短期 vs 长期**：在算力/数据受限的机器人项目中，人类先验往往是唯一可行起点。
- **样本效率**：纯 scaling 路线在真机数据稀缺时可能不经济；需与 sim2real、IL 预训练等结合。
- **可解释性与安全**：黑箱 scaling 策略在安全关键机器人上需额外验证层（参见 Sutton 更早的 *Verification, The Key to AI*）。

## 关联页面

- [Richard Sutton](../entities/richard-sutton.md)
- [Embodied Scaling Laws](./embodied-scaling-laws.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Model-Based RL](../methods/model-based-rl.md)
- [Data Flywheel](./data-flywheel.md)
- [From AGI to ASI](../entities/paper-from-agi-to-asi.md) — 宏观算力 scaling 对照

## 参考来源

- [The Bitter Lesson 原始资料](../../sources/blogs/sutton_bitter_lesson.md)
- [incompleteideas.net 一手资料索引](../../sources/sites/incompleteideas-net-rich-sutton.md)

## 推荐继续阅读

- [The Bitter Lesson 原文](http://incompleteideas.net/IncIdeas/BitterLesson.html)
- Sutton & Barto, *Reinforcement Learning: An Introduction* — TD/self-play 理论背景
- [Embodied Scaling Laws](./embodied-scaling-laws.md) — 机器人域微观 scaling 定律
