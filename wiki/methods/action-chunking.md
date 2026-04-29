---
type: method
tags: [imitation-learning, vla, action-chunking, latency, transformers, deployment]
status: complete
updated: 2026-04-29
summary: "Action Chunking 让策略一次输出未来多步动作序列，以降低长时序误差并缓解高延迟模型与高频控制器之间的时域错配。"
sources:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/humanoid_touch_dream.md
related:
  - ./behavior-cloning.md
  - ./humanoid-transformer-touch-dreaming.md
  - ./vla.md
  - ../queries/vla-deployment-guide.md
  - ../queries/vla-with-low-level-controller.md
  - ../tasks/bimanual-manipulation.md
---

# Action Chunking（动作块输出）

**Action Chunking**：让策略一次预测未来连续若干步动作，而不是每个控制周期只吐一帧动作。它最早在模仿学习和双臂操作场景里被广泛采用，现在也成为 VLA 与低层控制器结合时处理推理延迟的常见手段。

## 一句话定义

把“每一步都重新想”改成“先给一小段动作计划，再由执行层平滑落地”。

## 为什么重要

动作块的价值主要有两类：

1. **缓解长时序误差**：单步预测容易在时间上抖动，前一帧偏一点，后一帧继续偏，误差会累积。动作块把局部时域一起建模，更容易学到平滑一致的行为。
2. **缓解推理延迟**：当策略推理速度只有 5~20 Hz，而控制器需要 100~1000 Hz 时，动作块可以让低层在等待下一次推理结果时继续执行已有参考。

这也是 ACT、部分 diffusion policy，以及 VLA 真机部署里经常出现 chunk / horizon / buffer 设计的原因。

## 主要技术路线

与标准行为克隆只预测当前动作 $a_t$ 不同，动作块方法预测：

$$
\left[a_t, a_{t+1}, \dots, a_{t+K-1}\right]
$$

其中 $K$ 是 chunk 长度。部署时通常只执行其中前若干步，然后在下一个时刻用新 chunk 覆盖或拼接旧 chunk。

常见实现：

- **固定长度 chunk**：每次输出未来 4~32 步动作
- **重叠滚动执行**：每次只执行前半段，后半段被下一次预测覆盖
- **带 buffer 的异步执行**：策略线程低频更新 chunk，控制线程高频消费 chunk

## 和单步预测的区别

| 维度 | 单步动作预测 | Action Chunking |
|------|-------------|----------------|
| 输出形式 | 当前一步动作 | 未来多步动作序列 |
| 平滑性 | 容易抖动 | 更容易保持时序连续 |
| 长时序误差 | 容易 compounding error | 更稳，但不是彻底消除 |
| 延迟容忍 | 低 | 高 |
| 部署复杂度 | 低 | 需要 buffer / 覆盖策略 |

## 在机器人里的典型用途

### 1. 双臂模仿学习

双臂操作往往需要跨几百毫秒的协调，单步预测容易出现两臂不同步。动作块可以在一个时间窗里同时预测两臂未来动作，减少“左手已经到位，右手还没跟上”的时序问题。

### 2. VLA 真机部署

VLA 推理常有 50ms 以上延迟，因此不适合直接做高频闭环。更现实的做法是：

- VLA 输出 action chunk 或末端位姿 chunk
- 中间层做插值、限幅和安全过滤
- 低层控制器按高频消耗 chunk

### 3. 接触丰富任务

在插拔、擦拭、拧紧这类任务里，动作块有助于保持短时间内的动作一致性，避免策略因为每一帧独立采样而频繁切换接触意图。

[HTD](./humanoid-transformer-touch-dreaming.md) 把 action chunking 用在人形接触丰富型任务上：动作输出仍是短 horizon chunk，但训练时额外预测未来手部力和触觉 latent，减少 chunk 内“动作看似平滑但接触状态没学到”的问题。

## 设计时要注意什么

### Chunk 长度不能盲目变大

块越长，延迟容忍越高，但也越容易：

- 对环境变化反应慢
- 在 chunk 边界发生跳变
- 把错误动作持续执行更久

通常要根据任务频率和模型延迟选取：桌面操作常见 4~16 步，人形或高延迟 VLA 会配合更长 chunk 和更强的 fallback。

### 需要边界处理

最常见的问题不是 chunk 本身，而是两个 chunk 之间怎么切：

- 是否对首尾做插值
- 是否保留旧 chunk 后半段
- 是否允许新 chunk 立即覆盖旧计划

如果边界处理差，动作块会在切换瞬间产生明显抖动。

### 必须有 fallback

如果下一次推理结果迟到，系统不能直接空转。通常要有：

- 保持当前姿态
- 低速回零
- 打开夹爪 / 撤退
- 用上一段 chunk 的最后安全姿态继续执行

## 常见误区

- **误区 1：动作块等于规划。**  
  不完全是。它更像短时动作预测或执行缓冲，不等于全局任务规划。
- **误区 2：用了 chunk 就不会有 compounding error。**  
  只是在局部时域里更稳，长时间滚动执行仍会积累误差。
- **误区 3：chunk 越长越好。**  
  过长会削弱反馈，环境一变化就可能整段动作都过期。

## 参考来源

- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md) — ACT / ALOHA / action chunking 的核心背景
- [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — 生成式动作序列与长时域输出方式
- [sources/papers/humanoid_touch_dream.md](../../sources/papers/humanoid_touch_dream.md) — HTD 在人形接触丰富型操作中结合 action chunks 和 touch dreaming
- [Embodied-AI-Guide](../../sources/repos/embodied-ai-guide.md) — 具身智能能力栈与执行策略
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* — ACT 的代表性工作

## 关联页面

- [Behavior Cloning](./behavior-cloning.md) — 动作块是对单步 BC 的时间窗扩展
- [Humanoid Transformer with Touch Dreaming](./humanoid-transformer-touch-dreaming.md) — action chunks + 未来触觉 latent 预测的人形操作实例
- [Behavior Cloning Loss](../formalizations/behavior-cloning-loss.md) — 动作块模型（如 ACT）所优化的底层损失函数形式
- [VLA](./vla.md) — VLA 在真机部署时常结合 action chunking
- [ALOHA](../entities/aloha.md) — 经典的双臂遥操作硬件标杆
- [RoboTwin 2.0](../entities/robotwin.md) — 自动化数据生成平台
- [Query：VLA 真机部署指南](../queries/vla-deployment-guide.md) — 动作缓冲与异步执行
- [Query：VLA 与低级关节控制器融合架构](../queries/vla-with-low-level-controller.md) — VLA + WBC 的 action buffer 设计
- [Bimanual Manipulation](../tasks/bimanual-manipulation.md) — 双臂协调任务中常见 chunk 输出
