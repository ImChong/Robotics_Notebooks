---
type: query
tags: [vla, deployment, latency, manipulation, foundation-policy, real-robot]
status: complete
summary: "面向真机部署的 VLA 指南：如何处理推理延迟、动作缓冲、安全回退和数据回流。"
related:
  - ../methods/vla.md
  - ../concepts/foundation-policy.md
  - ../tasks/manipulation.md
  - ../tasks/loco-manipulation.md
  - ./control-architecture-comparison.md
sources:
  - ../../sources/papers/rl_foundation_models.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/sim2real.md
---

> **Query 产物**：本页由以下问题触发：「如何在真机上部署 VLA 策略？推理延迟怎么控制？」
> 综合来源：[VLA](../methods/vla.md)、[Foundation Policy](../concepts/foundation-policy.md)、[Manipulation](../tasks/manipulation.md)、[Loco-Manipulation](../tasks/loco-manipulation.md)、[控制架构综合对比](./control-architecture-comparison.md)

# Query：VLA 真机部署指南

## TL;DR 决策树

```text
你的 VLA 输出是什么？
├── 低级关节动作
│   ├── 推理延迟 > 50ms？
│   │   └→ 不要直接闭环驱动执行器；改为 action chunk + 低层控制器
│   └── 推理延迟 ≤ 20ms？
│       └→ 仍建议加 safety filter / rate limiter
└── 末端目标 / 技能 token
    └→ 更适合作为高层控制器，交给 WBC / impedance / skill library 执行
```

## 真机部署的核心原则

1. **把 VLA 当中高层策略，而不是 1kHz 电机控制器。**
2. **默认假设推理延迟不小，先围绕 latency 设计系统。**
3. **部署 pipeline 必须能把失败数据回流到训练集。**

## 典型系统切分

| 层级 | 推荐职责 |
|------|---------|
| VLA | 解析语言、理解场景、生成 action chunk / 末端子目标 |
| 中层执行 | 轨迹插值、抓取 primitive、技能状态机 |
| 低层控制 | impedance / PD / WBC / 力控，负责高频稳定执行 |

这类切分能显著缓解“大模型慢、机器人快”的时域错配问题。

## 如何处理推理延迟

### 1. 用 action chunk，而不是每步单发
VLA 每次输出未来 4~16 步动作或一个短时技能段，让控制器在 chunk 内平滑执行。这样即使推理需要 50ms+，机器人也不会每步都停住等模型。

### 2. 异步感知与推理
摄像头预处理、tokenization、GPU 推理和动作解码应并行流水线化，不要串行阻塞主控制线程。

### 3. 给低层控制器留兜底
如果新 chunk 迟到：
- 保持上一个目标的安全减速版本
- 切到 hold / retract / open-gripper fallback
- 对运动速度做限幅，避免过期命令直接打到执行器

## 部署 checklist

| 项目 | 为什么重要 |
|------|------------|
| 时间同步 | 图像和动作错 1~2 帧就会显著恶化策略 |
| 动作归一化一致性 | 训练/部署尺度不一致会直接导致爆动作 |
| 相机视角固定 | VLA 对视角分布很敏感 |
| 推理 profiling | 不测 P50/P95 latency，就无法稳定上线 |
| 安全边界 | workspace、力矩、速度、碰撞区必须硬限制 |
| 数据回流 | 真机失败片段是下一轮微调最有价值的数据 |

## 什么时候不该直接上 VLA

- 需要 >100Hz 的高频稳定控制
- 任务主要难点在接触力精度，而不是语义泛化
- 可用数据只有几十条，且分布极窄
- 现场没有 GPU 或稳定散热供电条件

## 推荐落地路线

1. **先离线评估**：固定回放数据上看成功率、动作平滑度和延迟
2. **再半实物 / 低速执行**：限速、限力、带急停
3. **部署 action chunk + 低层控制器**：不要直接替代执行层
4. **记录失败数据**：特别是遮挡、误抓、迟滞和长尾指令
5. **小步微调**：优先修最常见失败类型，而不是盲目加模型规模

## 常见坑

- **把 benchmark 上的成功率误当成真机 readiness**
- **忽略 P95 / P99 latency，只看平均值**
- **没有安全 fallback，模型一卡顿就继续执行旧命令**
- **数据清洗不够，训练日志和部署控制接口不一致**

## 一句话记忆

> VLA 真机部署的第一原则不是“模型多强”，而是“延迟可控、回退明确、动作可被传统控制层安全接住”。

## 参考来源

- [sources/papers/rl_foundation_models.md](../../sources/papers/rl_foundation_models.md) — RT-2 / π₀ / Octo 等 VLA 路线
- [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — 生成式动作模型与 chunked inference 背景
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md) — 真机部署与感知执行鸿沟的通用经验

## 关联页面

- [VLA](../methods/vla.md)
- [Foundation Policy（基础策略模型）](../concepts/foundation-policy.md)
- [Manipulation](../tasks/manipulation.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [Action Chunking（动作块输出）](../methods/action-chunking.md) — 缓解推理延迟和执行线程时域错配的常见手段
- [Query：VLA 与低级关节控制器融合架构](./vla-with-low-level-controller.md) — 中高层 VLA 如何接入 WBC / MPC / impedance
- [控制架构综合对比](./control-architecture-comparison.md)
