---
type: concept
tags: [systems-engineering, control, inference, realtime, deployment, vla]
status: complete
updated: 2026-11-07
related:
  - ./rtos-realtime-scheduling.md
  - ./llm-robotics-control-interfaces.md
  - ../entities/anthropic-embody.md
  - ../formalizations/control-loop-latency-modeling.md
  - ./edge-cloud-robotics.md
  - ../entities/dreamscale-labs.md
  - ../queries/vla-with-low-level-controller.md
  - ../queries/embodied-fm-taxonomy-loop.md
  - ../overview/hub-embodied-foundation-model.md
  - ../entities/jackhan-yobotics-e3-algorithm-template.md
  - ../overview/hub-systems-engineering.md
  - ../entities/paper-real-time-chunking.md
  - ../methods/action-chunking.md
  - ./flow-matching-embodied-policy.md
original:
  - ../../sources/papers/real_time_chunking_arxiv_2506_07339.md
  - ../../sources/sites/anthropic-claude-plays-robotics.md
  - ../../sources/papers/pi0_arxiv_2410_24164.md
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
  - ../../sources/papers/smolvla_arxiv_2506_01844.md
  - ../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md
sources:
  - ../../sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md
  - ../../sources/sites/anthropic-claude-plays-robotics.md
  - ../../sources/papers/real_time_chunking_arxiv_2506_07339.md
  - ../../sources/papers/pi0_arxiv_2410_24164.md
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
  - ../../sources/papers/smolvla_arxiv_2506_01844.md
  - ../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md
summary: "控制频率与推理频率解耦：高频执行环（PD/WBC）与低频策略/VLA 推理通过动作块、零阶保持或残差接口连接。"
---

# 控制频率与推理频率解耦

## 一句话定义

**控制/推理频率解耦** 让 **慢模型**（策略、VLA、规划）以低频产出动作意图，由 **快控制器**（PD、WBC、阻抗）以高频执行，避免把推理延迟写进力矩周期。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ZOH | Zero-Order Hold | 零阶保持，低频指令保持到下一次更新 |
| WBC | Whole-Body Control | 全身控制 |
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| PD | Proportional-Derivative | 比例–微分关节伺服 |
| Hz | Hertz | 频率单位 |

## 为什么重要

- 端到端网络常 10–50 Hz；平衡与力控需要 200–1000 Hz（本站工程经验区间，下列论文并未给出统一数字）。
- 强行同频要么欠稳定，要么 deadline miss——见 [延迟建模](../formalizations/control-loop-latency-modeling.md)。
- 论文给出的量级：RT-2 的 55B 模型约 **1–3 Hz**（5B 约 5 Hz），需放到多 TPU 云服务上查询；π₀ 的动作频率最高 50 Hz，4090 上单次推理 **73 ms**（观测前向 32 ms + 10 步动作前向 27 ms + 图像编码 14 ms）；GR00T N1 的 VLM（System 2）约 **10 Hz**、DiT 动作模块（System 1）约 **120 Hz**，采样 16 个动作 63.9 ms（L40）。
- [Embody](../entities/anthropic-embody.md) 给了一组厂商 API 量级：腿式实时约需 **83 Hz**，当前非推理 LLM 约 **0.2–0.4 Hz**（差约两个数量级）；物理环路在 10–125 Hz。他们在直接控制与代码控制的仿真里 **暂停物理** 才测出上界（固定基座机械臂没有平衡约束，未暂停）；这不是把聊天模型塞进力矩环的理由，而是 [LLM 控制接口](./llm-robotics-control-interfaces.md) 必须停在高层、由快控制器执行。

## 核心原理

常见接口：

1. **目标位形/轨迹**：低频更新 $q^*$，高频 PD 跟踪。
2. **动作块 / chunk**：一次推理输出 $H$ 步动作，执行器按控制周期弹出。
3. **残差叠加**：高频基座控制器 + 低频残差修正。
4. **异步队列**：推理线程写 lock-free 缓冲，控制线程只读最新。

```mermaid
flowchart LR
  sens[传感高频] --> ctrl[控制线程 500-1000Hz]
  sens --> infer[推理线程 10-50Hz]
  infer -->|q* / action chunk| buf[无锁缓冲]
  buf --> ctrl
  ctrl --> bus[总线力矩]
```

## 工程实践

### 论文给出的做法与数据

- **RTC（arXiv:2506.07339）：** 定义推理延迟 $d=\lfloor\delta/\Delta t\rfloor$（$\delta$ 为推理耗时，$\Delta t$ 为控制周期）。同步推理在 chunk 之间停顿，改变机器人动力学并造成分布偏移；朴素异步在 chunk 边界会跳变；异步需 $d\le H-s$。RTC 冻结前 $d$ 步并对其余重叠步做软约束的 inpainting，真机 π₀.₅（$H=50$、$\Delta t=20$ ms）延迟 +100/+200 ms 时吞吐几乎不降，两种时间集成（TE）因剧烈振荡触发保护性停机。代价：需通过去噪步反向传播，单步 14 → 35 ms。详见 [RTC](../entities/paper-real-time-chunking.md)。
- **SmolVLA 异步队列：** 动作队列剩余比例低于阈值 $g$ 就触发新推理，$g\ge E[\ell_S]/(n\Delta t)$ 可避免队列耗尽；用关节空间相似度过滤近重复观测；新旧队列在重叠段聚合。真机上单次任务 9.7 s（同步 13.75 s），60 s 内 19 次 vs 9 次，成功率相近。注意：RTC 是 inpainting 引导，SmolVLA 的聚合方式论文未细说，二者不是同一种方法。
- **chunk 长度与重规划频率的权衡（SmolVLA 消融，LIBERO）：** 块长 1/10/30/50/100 的平均成功率为 50.0/84.0/78.5/80.3/74.5；执行步数（更新观测前执行多少步）1/10/30/50 为 80.3/82.8/70.8/51.8，说明过长的开环执行显著伤害性能。RTC 论文也把“长执行视野降低反应性、短执行视野增加模式跳变”列为核心张力。

### 本站补充

- 例：Yobotics E3 模板约 **50 Hz 推理 / 500 Hz 发令**——见 [算法模板实体](../entities/jackhan-yobotics-e3-algorithm-template.md)。
- VLA + 底层控制器模式见 [VLA 与底层控制](../queries/vla-with-low-level-controller.md)；选型语境见 [具身大模型分类学选型闭环](../queries/embodied-fm-taxonomy-loop.md)。
- 推理超时：保持上一动作或切入安全 FSM，禁止阻塞控制线程。
- 度量：推理延迟、队列年龄、跟踪误差与控制 miss。

## 局限与风险

- Chunk 过长会「开环太久」：SmolVLA 在 LIBERO 上每次执行 50 步才更新观测时成功率只有 51.8%（每步更新 80.3%）；接触丰富任务上的影响更大属于合理推测，需自行验证。
- 论文多在位置控制的操作臂上验证；RTC 论文自己指出腿足等更动态场景只在仿真里测过，没有真机结果。
- 推理期方案会增加计算：RTC 需穿过去噪步反向传播；SmolVLA 的异步需要网络与近重复过滤策略。
- 两边状态不同步（观测时刻 vs 执行时刻）会造成分布偏移。

## 关联页面

- [Dreamscale Labs — 云端机器人基础模型推理](../entities/dreamscale-labs.md) — 官方 SDK 和 HUD 集成展示模型云端推理、动作块与本地执行/安全边界。

- [RTOS 与实时调度](./rtos-realtime-scheduling.md)
- [边缘–云端协同](./edge-cloud-robotics.md)
- [机器人安全状态机](./robot-safety-state-machine.md)
- [LLM 机器人控制接口](./llm-robotics-control-interfaces.md)
- [Embody](../entities/anthropic-embody.md)
- [Real-Time Chunking](../entities/paper-real-time-chunking.md)
- [Action Chunking](../methods/action-chunking.md)

## 参考来源

- [RTC 论文归档（arXiv:2506.07339）](../../sources/papers/real_time_chunking_arxiv_2506_07339.md)
- [π₀ 论文归档](../../sources/papers/pi0_arxiv_2410_24164.md)
- [GR00T N1 论文归档](../../sources/papers/gr00t_n1_arxiv_2503_14734.md)
- [SmolVLA 论文归档](../../sources/papers/smolvla_arxiv_2506_01844.md)
- [RT-2 论文归档](../../sources/papers/rt2_vla_reading_arxiv_2307_15818.md)
- [DDS/RTOS/边云/OTA/安全 FSM 一手资料](../../sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md)
- [Claude plays robotics 归档](../../sources/sites/anthropic-claude-plays-robotics.md)

## 推荐继续阅读

- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)
- [Real-Time Execution of Action Chunking Flow Policies（arXiv:2506.07339）](https://arxiv.org/abs/2506.07339)
- [Anthropic：Claude plays robotics](https://www.anthropic.com/research/claude-plays-robotics)
