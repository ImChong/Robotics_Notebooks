---
type: method
tags: [vla, fleet-scale-rl, offline-to-online-rl, post-training, data-flywheel, agibot, flow-matching]
status: complete
updated: 2026-05-06
related:
  - ./vla.md
  - ./π0-policy.md
  - ./reinforcement-learning.md
  - ./policy-optimization.md
  - ../concepts/foundation-policy.md
  - ../concepts/data-flywheel.md
  - ../comparisons/online-vs-offline-rl.md
sources:
  - ../../sources/papers/lwd.md
summary: "LWD 是 AGIBOT 提出的车队级 offline-to-online RL 后训练框架，通过 DIVL + QAM 把部署中的异构经验（成功/失败/人为干预）转化为单一 VLA generalist 策略的持续改进。"
---

# LWD（Learning while Deploying）

**LWD（Learning while Deploying）** 是 AGIBOT Research 在 2026 年提出的**车队级（fleet-scale）offline-to-online 强化学习后训练框架**，专门用于让一个**通用 VLA 策略**在部署中持续自我改进，而不是把"部署"看成训练的终点。

## 一句话定义

把一群真实机器人放出去做事，再把它们做出来的"成功 / 失败 / 半成 / 救场 / 人为接管"全部喂回同一个 RL 学习器，让一个 generalist VLA 策略一边跑一边变强。

## 为什么重要

- 现有 VLA / [Foundation Policy](../concepts/foundation-policy.md) 在实验室预训练得很强，但**真实家庭/门店/工厂的分布永远在漂**：台面高度、水果形状、透明摇酒壶……固定测试集解决不了这件事。
- 业内的[数据飞轮（Data Flywheel）](../concepts/data-flywheel.md)思路偏向"部署 → 抽好动作 → 模仿学习"，本质还是把部署当成"高质量演示的来源"。LWD 改写了这个假设：**部署产出的不只是好动作，还有失败、半成、救场、干预——这些在 RL 视角下都是有用信号**。
- 对**单一 generalist 策略 × 多任务车队**这一组合，过去工作要么训 specialist，要么停留在仿真，LWD 是少见的**真机车队 + 单一通用策略 + RL 闭环**实证。

## 主要技术路线

```
预训练 VLA 策略
        ↓ offline RL 初始化
策略 + critic（用专家演示 / 历史 rollout / 失败模式探索数据初始化）
        ↓ 部署到机器人车队
车队执行任务，产生（成功 / 失败 / 干预）轨迹
        ↓ 上传到共享 online replay buffer
集中式学习器：在 offline buffer + online buffer 上做统一 RL 更新
        ↓ 周期性 push 新 checkpoint
车队继续部署 → 更多异构经验 → 更新 …
```

整个闭环的关键是 **offline 与 online 阶段共用同一个 RL 学习器和同一份目标函数**——offline 提供稳定底盘，online 暴露真实部署分布，但两边不切换算法、不切换损失。LWD 称之为"**offline-to-online RL 数据飞轮**"。

## 核心算法组件

LWD 在车队级 RL 上面对两个具体困难，并各自给出一个组件作答：

### 1. DIVL — Distributional Implicit Value Learning

**问题**：车队 replay 极度异构（不同任务、不同 horizon、不同奖励稀疏度、不同程度的人为干预），还要在持续变化的混合数据上学价值。直接拟合一个标量 value target 会不稳定，长程任务上 credit assignment 也很难（一杯功夫茶最后倒不进去，可能是几分钟前那一抓没抓稳）。

**做法**：

- 不再回归一个标量 Q，而是学习一个**关于数据分布内动作-价值的分布**，再从分布中取**分位数统计量**作为 TD bootstrap 目标。
- 保留 [IQL](../comparisons/online-vs-offline-rl.md) 那种"只在数据分布内做价值学习、避免对 OOD 动作做最大化"的核心思想，但用分布建模容纳异构 replay 的方差，缓解过估计。
- 长程任务进一步使用**多步 TD 目标**，把稀疏的终态奖励更高效地往回传。

### 2. QAM — Q-learning with Adjoint Matching

**问题**：现代 VLA 普遍使用 flow-matching / diffusion 这类**多步生成式动作头**（[π₀](./π0-policy.md)、[Diffusion Policy](./diffusion-policy.md) 等）。这类策略的动作 likelihood 不好算，直接对整个生成过程做 critic-gradient 反向传播也不稳定，让经典 likelihood-based RL 算法很难直接套上。

**做法**：

- 用 critic 梯度引导**沿 flow 生成轨迹的局部回归**来抽取策略，而不是对完整生成链做端到端反传。
- 这样既不需要可解析的动作 likelihood，也避开了多步 backprop 的稳定性问题。

DIVL 与 QAM 一起实现了**策略评估（value learning）与策略抽取（policy extraction）的解耦**：DIVL 负责从异构 offline+online replay 中学价值，QAM 把这些价值估计稳定地落到 flow-based VLA 策略上。

## 与已有路线的差异

| 维度 | 模仿式数据飞轮（Imitation-based） | LWD（RL-based） |
|------|----------------------------------|-----------------|
| 部署的角色 | 抽取好动作的来源 | 直接进入训练循环 |
| 用得上的轨迹 | 主要是成功 / 高质量轨迹 | 成功 / 失败 / 半成 / 干预 / 救场都用 |
| 学习信号 | 行为模仿 | 奖励 / 价值 |
| 跨任务 | 通常按任务专家分别做 | 单一 generalist 策略覆盖多任务车队 |
| 离线-在线 | 两阶段不同算法 / pipeline | 同一 RL 学习器统一离线与在线更新 |
| 适合 horizon | 中短程 | 长程任务收益尤其明显 |

## 实验设置（关键事实）

- **平台**：Agibot G1 双臂机器人车队。
- **任务**：8 个真实操作任务，含**长程任务**（3–5 分钟，如功夫茶、调酒、果汁、装鞋）和**超市补货**等。
- **策略**：单一 generalist VLA，跨所有任务联合训练，不为每个任务训单独控制器。
- **结果**：相对前代后训练 baseline，LWD 在**成功率**上整体提升，**长程任务**的提升尤其显著；同时**平均周期时间（cycle time）下降**，说明 offline-to-online 后训练不仅让策略"更可能成功"，也让"完成方式更高效"。

## 适合放在系统中的哪一层

- **预训练之后的持续后训练阶段**：上游是用海量数据做完 IL/SFT 的 [VLA](./vla.md)；LWD 接管"上线之后还要继续变强"的部分。
- **车队级基础设施**：需要一套能可靠回传 rollout、记录人为干预、做集中式训练并把 checkpoint 推回去的工程链路；和[数据飞轮](../concepts/data-flywheel.md)的工程要求高度重叠，但学习目标从模仿换成了 RL。
- **不适合直接替换** 1 kHz 关节控制层；它优化的是 generalist 策略层，底层执行仍然由相应的低级控制器和 [Safety Filter](../concepts/safety-filter.md) 兜底。

## 常见误区

- **误区 1：LWD 等于"在线微调"。** 它的 offline 与 online 阶段共用一个 RL 学习器和目标，是一个统一框架，而不是"先 IL 再加点 online RL"的简单拼接。
- **误区 2：LWD = 数据飞轮的换皮。** 区别是把"部署 → 提取好动作 → 模仿"换成"部署 → 全谱经验 → RL 更新"，并明确把失败和人为干预算进训练信号。
- **误区 3：随便一个 VLA 都能直接接 LWD。** Flow-based 动作头的策略抽取本身就是难点，LWD 用 QAM 才把这件事做稳；换成其他动作头时，对应的策略抽取方法需要重新对齐。

## 参考来源

- [sources/papers/lwd.md](../../sources/papers/lwd.md) — 论文 ingest 档案
- AGIBOT Research, *Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies* (2026) — [项目页](https://finch.agibot.com/research/lwd) / [PDF](https://finch-static.agibot.com/LWD/lwd-paper.pdf)
- 背景延伸：Kostrikov et al., *Implicit Q-Learning* (2021)；Black et al., *π₀: A Vision-Language-Action Flow Model for General Robot Control* (2024)

## 关联页面

- [VLA (Vision-Language-Action)](./vla.md) — LWD 改进的对象就是一个 VLA generalist 策略
- [π₀ (Pi-zero) 策略模型](./π0-policy.md) — Flow-matching 动作头的代表，QAM 正是为这类策略设计
- [Foundation Policy](../concepts/foundation-policy.md) — LWD 是基础策略"持续后训练"阶段的具体方案
- [Data Flywheel（数据飞轮）](../concepts/data-flywheel.md) — LWD 对应的是"RL 版本"的数据飞轮
- [Online RL vs Offline RL](../comparisons/online-vs-offline-rl.md) — LWD 的 offline-to-online 定位
- [Reinforcement Learning](./reinforcement-learning.md) — RL 基础范式
- [Diffusion Policy](./diffusion-policy.md) — 与 flow-matching 同属生成式动作头家族

## 推荐继续阅读

- AGIBOT Research, [LWD 项目页](https://finch.agibot.com/research/lwd)
- Kostrikov et al., *Offline Reinforcement Learning with Implicit Q-Learning* — DIVL 的"implicit value learning"出处
- Lipman et al., *Flow Matching for Generative Modeling* — flow-based 动作头背后的生成式建模基础
