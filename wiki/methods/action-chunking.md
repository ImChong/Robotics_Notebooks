---
type: method
tags: [imitation-learning, vla, action-chunking, latency, transformers, deployment]
status: complete
updated: 2026-09-05
summary: "Action Chunking 让策略一次输出未来多步动作序列，以降低长时序误差并缓解高延迟模型与高频控制器之间的时域错配；机制上可拆为延迟观测条件化与隐式集成，部署不必等于播放整段 chunk；长 open-loop 执行多因短上下文模仿非马尔可夫专家。"
sources:
  - ../../sources/repos/act-aloha.md
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/humanoid_touch_dream.md
  - ../../sources/repos/xiaomi-robotics-0.md
  - ../../sources/papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md
  - ../../sources/papers/why_action_chunking_improves_bc_corl2026.md
  - ../../sources/papers/spd_corl_2026.md
  - ../../sources/papers/wam_realtime_async_arxiv_2608_01880.md
  - ../../sources/courses/sergey_levine_diffusion_rl_robotics_simons_youtube.md
  - ../../sources/papers/nestdex_arxiv_2608_13362.md
  - ../../sources/papers/revisiting_open_loop_action_chunking_arxiv_2608_15938.md
  - ../../sources/papers/arli_arxiv_2608_23831.md
  - ../../sources/papers/video2door_traversal_arxiv_2608_20251.md
  - ../../sources/blogs/seohong_behavioral_cloning_mystery.md
related:
  - ../entities/paper-act.md
  - ./behavior-cloning.md
  - ./humanoid-transformer-touch-dreaming.md
  - ./vla.md
  - ../entities/paper-flashvla.md
  - ../entities/xiaomi-robotics-0.md
  - ../queries/vla-deployment-guide.md
  - ../queries/vla-with-low-level-controller.md
  - ../tasks/bimanual-manipulation.md
  - ../entities/paper-taco-tactile-sensor-benchmark.md
  - ../entities/paper-chronos.md
  - ../entities/paper-pi-r2.md
  - ../entities/paper-why-action-chunking-improves-bc.md
  - ../entities/paper-spd.md
  - ../entities/paper-autointervene.md
  - ../entities/paper-nestdex.md
  - ../entities/paper-wam-realtime-async.md
  - ../entities/paper-glancewam.md
  - ../entities/paper-tonav.md
  - ../entities/paper-arli.md
  - ../entities/paper-smoothrl.md
  - ../entities/paper-video2door-traversal.md
  - ../entities/paper-revisiting-open-loop-action-chunking.md
  - ../concepts/behavioral-cloning-mysteries.md
  - ../overview/sergey-levine-diffusion-expressive-policies.md
---

# Action Chunking（动作块输出）

**Action Chunking**：让策略一次预测未来连续若干步动作，而不是每个控制周期只吐一帧动作。它最早在模仿学习和双臂操作场景里被广泛采用，现在也成为 VLA 与低层控制器结合时处理推理延迟的常见手段。

## 一句话定义

把“每一步都重新想”改成“先给一小段动作计划，再由执行层平滑落地”。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| ACT | Action Chunking Transformer | 预测动作块的序列模型架构，常与 ALOHA 配套 |
| AI | Artificial Intelligence | 人工智能 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| BC | Behavior Cloning | 将状态映射到动作的监督式模仿，易受分布偏移影响 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |

## 为什么重要

动作块的价值主要有两类：

1. **缓解长时序误差**：单步预测容易在时间上抖动，前一帧偏一点，后一帧继续偏，误差会累积。动作块把局部时域一起建模，更容易学到平滑一致的行为。
2. **缓解推理延迟**：当策略推理速度只有 5~20 Hz，而控制器需要 100~1000 Hz 时，动作块可以让低层在等待下一次推理结果时继续执行已有参考。

这也是 ACT、部分 diffusion policy，以及 VLA 真机部署里经常出现 chunk / horizon / buffer 设计的原因。

从动作分布一侧看，[Levine @ Simons 2026](../overview/sergey-levine-diffusion-expressive-policies.md) 把「敢输出很长的 action chunk」归因于 **扩散 / flow 等高表达力动作头**——没有足以覆盖高维多模态序列的分布族，长 chunk 在实践上难以学稳；这与下方「部署协议」讨论互补：表达力解释**为何能训长序列**，Delay / RDE 解释**部署时是否必须整段播放**。

机制上要分开三件事：**训练时拟合动作块**、**部署时是否连续播放该块**、以及 **开环执行前缀有多长（execution horizon）**。CoRL 2026 的 [Why Action Chunking Improves BC](../entities/paper-why-action-chunking-improves-bc.md) 表明：相对单步 BC 的成功率跃迁，主因更接近 **用过去观测预测动作（delayed policy）** 与 **同时学多种时延关系带来的隐式集成**；「时序一致性 / 有效地平线缩短 / 训长 chunk 的表征红利」不足以单独解释。同一 \(\hat\pi_k\) 可用 **Randomized Delay Ensemble（RDE）** 在多数设定匹配标准 chunk 执行——训练目标与执行协议可以解耦。

[Revisiting Open-Loop Execution](../entities/paper-revisiting-open-loop-action-chunking.md)（MIT / Berkeley，arXiv:2608.15938）进一步把 **长 open-loop execution horizon** 归因于 **短上下文（常见 \(T_o=1\)–\(2\)）策略模仿非马尔可夫专家**：复合误差有影响，但通常弱于专家隐状态不可观；**加长观测上下文**（如 8–20 帧）+ **double encoder** 可让 \(T_{\mathrm{exec}}^*\rightarrow 1\) 的 **完全 reactive** 策略在数据充足时优于短上下文长开环执行——与 Why AC 的「不必播完整 chunk」形成 **execution horizon ↔ context length** 互补轴。

[BC Mysteries](../concepts/behavioral-cloning-mysteries.md) 给出第三条证据：在人类风格（窄、时间相关）数据上，**无限数据** 的纯闭环 \(\pi(a_t\mid s_t)\) 可以完全失败，而 length-25 开环能做；把过去 24 帧状态拼进闭环 **并不** 自动追上开环（因果混淆 / 输入空间更大）。与 Revisiting 合并读：开环是短记忆补丁；要让闭环赢，上下文必须编码 **专家隐状态**（接触意图、分段决策），而不是更长的关节角窗口。

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

[Xiaomi-Robotics-0](../entities/xiaomi-robotics-0.md) 把 **异步 chunk** 推进一步：在 **推理耗时 Δtinf** 内仍执行当前 chunk 的未消费步，并对下一 chunk 用 **已提交动作前缀** 条件化；训练侧用 **Λ 形注意力、前缀随机遮蔽与损失重加权** 减轻「模型抄前缀、弱化视觉反馈」的捷径行为（细节以论文为准）。

### 3. 接触丰富任务

在插拔、擦拭、拧紧这类任务里，动作块有助于保持短时间内的动作一致性，避免策略因为每一帧独立采样而频繁切换接触意图。

[HTD](./humanoid-transformer-touch-dreaming.md) 把 action chunking 用在人形接触丰富型任务上：动作输出仍是短 horizon chunk，但训练时额外预测未来手部力和触觉 latent，减少 chunk 内“动作看似平滑但接触状态没学到”的问题。

[SPD](../entities/paper-spd.md) 在 56-DoF 双臂灵巧手上给出互补证据：单帧条件时 8 步 chunk 会抖崩，π0 风格的单帧 + 32 步也明显弱；加上 **32 步 visuomotor 历史** 后，短 8 步 chunk 反而最强——时序一致性来自上下文，反应来自短计划。该配置也是仿真预训练收益最大的一档（平均进度 +18 点）。

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
  只是在局部时域里更稳，长时间滚动执行仍会积累误差；理论尺度上 AC 与 delayed policy 同为 \(\mathcal{O}((k+1)^{H/k}\epsilon)\)，并不能神奇消除指数复合（见 [Why AC Improves BC](../entities/paper-why-action-chunking-improves-bc.md)）。
- **误区 3：chunk 越长越好。**  
  过长会削弱反馈，环境一变化就可能整段动作都过期。
- **误区 4：成功了就一定是「时序一致性」或「有效地平线变短」。**  
  对照实验里，只条件于过去观测的 Delay / 随机时延 RDE 常能匹配标准 AC；不要把部署协议误当成唯一因果。

## 参考来源

- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md) — ACT / ALOHA / action chunking 的核心背景
- [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — 生成式动作序列与长时域输出方式
- [sources/papers/humanoid_touch_dream.md](../../sources/papers/humanoid_touch_dream.md) — HTD 在人形接触丰富型操作中结合 action chunks 和 touch dreaming
- [sources/repos/xiaomi-robotics-0.md](../../sources/repos/xiaomi-robotics-0.md) — Xiaomi-Robotics-0：异步 VLA chunk 条件化与后训练技巧归档
- [Embodied-AI-Guide](../../sources/repos/embodied-ai-guide.md) — 具身智能能力栈与执行策略
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* — ACT 的代表性工作
- [sources/papers/defi_arxiv_2604_16391.md](../../sources/papers/defi_arxiv_2604_16391.md) — DeFI：2D 视频预测与 3D 动作推理拆分预训练，扩散适配器输出动作 chunk
- [sources/papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md](../../sources/papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md) — TacO：ACT + 模态特异触觉编码器的跨传感器基准
- [sources/papers/why_action_chunking_improves_bc_corl2026.md](../../sources/papers/why_action_chunking_improves_bc_corl2026.md) — CoRL 2026：chunk 收益机制（Delay / RDE / 隐式集成）
- [SPD 论文归档](../../sources/papers/spd_corl_2026.md) — 历史条件化使短 chunk 可反应（CoRL 2026）
- [sources/courses/sergey_levine_diffusion_rl_robotics_simons_youtube.md](../../sources/courses/sergey_levine_diffusion_rl_robotics_simons_youtube.md) — Levine：扩散/flow 使大块 action chunk 成为连续控制默认接口

## HMI 开源主表入口

[ACT](https://github.com/tonyzhaozh/act) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)。

主表定位：条件 VAE + Transformer 一次预测动作块，时间集成平滑控制；低成本双臂采数与真机部署代码，是模仿学习常用基线。本库以本方法页为 ACT 详情节点，不另建 `act-aloha` 实体。

覆盖核对见 [HMI 开源项目主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)。

## 关联页面

- [Behavior Cloning](./behavior-cloning.md) — 动作块是对单步 BC 的时间窗扩展
- [Humanoid Transformer with Touch Dreaming](./humanoid-transformer-touch-dreaming.md) — action chunks + 未来触觉 latent 预测的人形操作实例
- [Behavior Cloning Loss](../formalizations/behavior-cloning-loss.md) — 动作块模型（如 ACT）所优化的底层损失函数形式
- [VLA](./vla.md) — VLA 在真机部署时常结合 action chunking
- [Xiaomi-Robotics-0](../entities/xiaomi-robotics-0.md) — 开源 VLA 的异步 chunk rollout 与后训练实例
- [ALOHA](../entities/aloha.md) — 经典的双臂遥操作硬件标杆
- [RoboTwin 2.0](../entities/robotwin.md) — 自动化数据生成平台
- [Query：VLA 真机部署指南](../queries/vla-deployment-guide.md) — 动作缓冲与异步执行
- [Query：VLA 与低级关节控制器融合架构](../queries/vla-with-low-level-controller.md) — VLA + WBC 的 action buffer 设计
- [Bimanual Manipulation](../tasks/bimanual-manipulation.md) — 双臂协调任务中常见 chunk 输出
- [TacO（触觉传感器操作基准）](../entities/paper-taco-tactile-sensor-benchmark.md) — ACT 作为跨模态触觉硬件评测骨干（chunk 64 / 执行 32）
- [ParcelStow](../entities/paper-parcelstow.md) — ACT 在 G1 包裹插入上的时间鲁棒性对照：标称 100%，\(r=2\) 仅 53%
- [Chronos](../entities/paper-chronos.md) — 把 action chunk 当广义坐标，经 IMLE 先验 + 二阶加速度桥精炼（arXiv:2606.30318）
- [πR²](../entities/paper-pi-r2.md) — 对 chunking flow 做本体感快通道 + 时延自适应日程，GR00T 约 25 Hz 闭环（arXiv:2607.26055）
- [Why Action Chunking Improves BC](../entities/paper-why-action-chunking-improves-bc.md) — CoRL 2026：Delay / RDE 机制消融与「训练≠必须 chunk 执行」
- [FlashVLA](../entities/paper-flashvla.md) — 流匹配 VLA 用交错噪声缓冲把 chunk 解码摊到时间轴上，修异步错配（arXiv:2608.27384）
- [SmoothRL](../entities/paper-smoothrl.md) — 异步执行环内对冻结 VLA 做 value-gradient 在线 RL（arXiv:2608.29768；项目页已上线、仍未开源）
- [GlanceWAM](../entities/paper-glancewam.md) — WAM 动作块在潜空间 48 ms 解码，想象异步离环（arXiv:2608.23927）
- [TONAV](../entities/paper-tonav.md) — 位置–速度动作块稳住四足铰接接触（arXiv:2608.22296）
- [Revisiting Open-Loop Execution](../entities/paper-revisiting-open-loop-action-chunking.md) — arXiv:2608.15938：长 execution horizon 多因短上下文；够长 \(T_o\) 后 reactive 最优
- [BC Mysteries](../concepts/behavioral-cloning-mysteries.md) — 真机风格数据上开环 BC 可完胜逐步闭环；乱加历史不等于修好
- [SPD](../entities/paper-spd.md) — 灵巧真机：历史窗才能把 chunk 缩到 8 步且吃到仿真预训练（CoRL 2026）
- [AutoIntervene](../entities/paper-autointervene.md) — 对提议 chunk 做视觉–动作支持监控与双向自动接管（arXiv:2608.07065）
- [NestDex](../entities/paper-nestdex.md) — 内外层均用 chunk + 时间集成；瓶抓消融显示闭环适应接触、ensemble 降 jerk（arXiv:2608.13362）
- [ReflexVLA](../entities/paper-reflexvla.md) — 动态任务上钉死「大 chunk + 短执行地平线」；异步 chunk=8 / horizon=2（arXiv:2608.14379）
- [ARLI](../entities/paper-arli.md) — 异步 chunk 执行下延迟感知 RL 后训练；中间已承诺动作条件 DSRL（arXiv:2608.23831；确认未开源）
- [Video2DoorTraversal（论文实体）](../entities/paper-video2door-traversal.md) — ArticuACT 在 ACT chunk=100 上加机器人系 Plücker 与交互进度辅助头（arXiv:2608.20251；代码待发布）
- [WAM 实时异步部署](../entities/paper-wam-realtime-async.md) — 双臂 WAM 上对照 sync/async/blend/simple/infer/train（arXiv:2608.01880）
- [Sergey Levine：表达力更强的连续动作策略](../overview/sergey-levine-diffusion-expressive-policies.md) — 生成式动作头如何使长 chunk 在实践上可行
