---
type: method
tags: [vla, vision-language-action, foundation-policy, manipulation, rt2, pi0]
status: complete
updated: 2026-04-20
summary: "VLA 把语言、视觉和动作统一进一个多模态策略模型，使机器人能够直接从自然语言与图像条件生成控制动作。"
related:
  - ../concepts/foundation-policy.md
  - ./behavior-cloning.md
  - ./action-chunking.md
  - ./diffusion-policy.md
  - ../tasks/manipulation.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/rl_foundation_models.md
  - ../../sources/papers/diffusion_and_gen.md
---

# VLA（Vision-Language-Action）

**VLA**：把视觉、语言和机器人动作统一到同一个模型里，让策略不只“看见状态后输出动作”，还能够显式理解任务指令和语义约束。

## 一句话定义

VLA 可以看成机器人版的多模态 foundation model：输入“看到了什么 + 要做什么”，输出“下一步怎么动”。

## 为什么重要

- 它把“任务描述”从手写 reward 或手工 state machine，转成自然语言接口。
- 它是 RT-2、π₀、OpenVLA、Octo 一类通用操作策略的共同抽象。
- 它让一个模型处理多任务成为可能，但代价是更大的数据需求、更高推理延迟，以及更复杂的部署链路。

## 典型架构

```text
语言指令 + 图像 / proprioception
          ↓
  多模态编码器（VLM / Transformer）
          ↓
  动作解码器（token / chunk / flow / diffusion）
          ↓
     末端位姿 / 关节动作
```

常见实现：
- **RT-2**：把 web-scale VLM 能力迁移到机器人控制
- **π₀**：在 VLA 上加入 Flow Matching，生成连续动作序列
- **OpenVLA / Octo**：更强调开源数据、跨任务泛化和 fine-tune 流程

## VLA 与传统策略的区别

| 维度 | 传统 BC / RL 策略 | VLA |
|------|-------------------|-----|
| 任务输入 | 预定义 observation / goal | 自然语言 + 视觉 + 状态 |
| 泛化方式 | task-specific | 多任务/零样本/少样本 |
| 数据规模 | 百到千级演示 | 通常需要数千到数十万演示 |
| 推理开销 | 低，适合高频控制 | 高，常见 50ms+，需异步部署 |
| 适合任务 | 单任务控制 | 通用操作、多任务调度 |

## 核心优势

### 1. 语言条件化
可以直接用“把红色杯子放到左边托盘”之类的任务描述驱动策略，而不是单独写状态机。

### 2. 多任务统一
VLA 常把抓取、放置、开关门、抽屉操作等任务放进一个统一模型，而非每项任务单训一个 policy。

### 3. 语义泛化
Web 知识和视觉语义可以帮助机器人处理训练集中稀疏出现的物体、关系和指令表述。

## 工程瓶颈

### 1. 推理延迟
VLA 通常不是高频底层控制器，真机上常见 50ms 以上推理延迟，因此更适合输出 action chunk、目标位姿或中频命令，再由低层控制器执行。

### 2. 数据规模要求高
想要稳健泛化，通常需要大量多样化演示数据。十几条示教可以做 task-specific BC，但远不足以支撑通用 VLA。

### 3. 部署链路复杂
摄像头时间同步、图像预处理、prompt 模板、动作反归一化、GPU 推理和安全 fallback，任何一步都可能拖垮真机体验。

## 适合放在系统中的哪一层

- **高层任务规划 / 中层动作生成**：适合
- **1kHz 力矩闭环控制**：通常不适合
- **和 WBC / impedance / skill library 结合**：当前更现实的真机方案
- **常见落地方式**：输出 [Action Chunking](./action-chunking.md) 或末端目标，再交给低层控制器和 [Safety Filter](../concepts/safety-filter.md) 执行

## 常见误区

- **误区 1：VLA 的实时性和传统控制器相当。**
  通常并非如此，必须认真处理推理频率和动作缓冲。
- **误区 2：VLA 可以在十条演示上学成通用能力。**
  通用能力依赖大规模、异构、多任务数据。
- **误区 3：VLA = 直接替代所有控制模块。**
  当前更可靠的工程做法仍是“VLA 负责语义与任务层，传统控制负责执行层”。

## 参考来源

- [sources/papers/rl_foundation_models.md](../../sources/papers/rl_foundation_models.md) — RT-1 / RT-2 / π₀ / Octo / TD-MPC2 综述
- [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — π₀ 与生成式动作建模路线
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*
- Black et al., *π₀: A Vision-Language-Action Flow Model for General Robot Control*

## 关联页面

- [Foundation Policy（基础策略模型）](../concepts/foundation-policy.md)
- [Manipulation](../tasks/manipulation.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [Action Chunking](./action-chunking.md)
- [Diffusion Policy](./diffusion-policy.md)
- [Behavior Cloning](./behavior-cloning.md)
- [Query：VLA 真机部署指南](../queries/vla-deployment-guide.md)
- [Query：VLA 与低级关节控制器融合架构](../queries/vla-with-low-level-controller.md)
- [Safety Filter](../concepts/safety-filter.md)

## 推荐继续阅读

- RT-2 / π₀ 原论文或项目博客
- OpenVLA / Octo 开源实现
- [Query：如何在真机上部署 VLA 策略？](../queries/vla-deployment-guide.md)
- [Query：VLA 与低级关节控制器融合架构](../queries/vla-with-low-level-controller.md)
