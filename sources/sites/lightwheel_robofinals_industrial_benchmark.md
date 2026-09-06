# RoboFinals Industrial Benchmark（光轮媒体文）

> 来源归档

- **标题：** RoboFinals Industrial Benchmark — How Early Adopters Are Scaling Model Evaluation for Physical AI
- **类型：** site（厂商媒体 / 技术叙事）
- **来源：** 光轮科技（Lightwheel）
- **链接：** https://lightwheel.ai/media/robofinals-industrial-benchmark
- **主发布页：** https://lightwheel.ai/robofinals
- **入库日期：** 2026-09-06
- **一句话说明：** RoboFinals 工业 benchmark 深度文：评测成 Physical AI 瓶颈、RoboFinals-100 结构、**AutoDataGen** 合成数据管线、Isaac Lab-Arena + **NVIDIA OSMO** 编排栈、早期采用方（Qwen/Fourier/RoboForce/Peritas）。
- **沉淀到 wiki：** [`wiki/entities/lightwheel-robofinals.md`](../../wiki/entities/lightwheel-robofinals.md)

---

## 核心论点

- **Training builds capability. Evaluation defines progress.** 前沿模型已超越多数学术仿真榜，真机评测不可扩展，评测成为主瓶颈。
- 目标：成为 robotics 的 **「ImageNet」**——共享基础设施，让团队可测量进展、对比系统。

## RoboFinals-100（与发布页一致）

- 渐进难度、高任务多样性、行业对齐真实感。
- 家庭 / 工厂 / 零售；刚体 + 铰接 + 可变形；跨桌面臂 / 移动操作 / loco-manipulation。

## AutoDataGen（合成动作数据）

光轮在 Isaac Lab 上开发的 **自动化仿真数据生成** 管线（**未列公开仓库**，截至 2026-09-06）：

| 能力 | 说明 |
|------|------|
| LLM 任务分解 | 从任务代码、场景配置或自然语言拆成原子技能 |
| Isaac Lab 集成 | 作为附加包，最小侵入现有项目 |
| 统一抽象 | 全流程一致接口，可扩展复用 |
| 可插拔模块 | 自定义 decomposer / skill / action adapter 注册 |
| LW-BenchHub 联动 | 自动分解并执行 benchmark 任务 |

用途：在跑 RoboFinals 评测前，为 benchmark 任务 **自动生成动作数据**。

## RoboFinals Evaluation Stack

```text
SimReady 环境 + RoboFinals-100 任务
        ↓
Isaac Lab-Arena（环境 / 机器人 / 任务解耦）
        ↓
Lightwheel 扩展：复杂任务逻辑、长程工作流、通才模型评测协议
        ↓
NVIDIA OSMO（分布式编排：实验、调度、并行 rollout）
        ↓
云 GPU（含 Nebius 集群）→ 数千 episode 并行
```

- **Isaac Lab-Arena**：GitHub 开源；评测与任务层与光轮联合设计。
- **OSMO**：NVIDIA 分布式 AI 工作负载编排。

## 早期采用方（文内列举）

| 团队 | 用途 |
|------|------|
| Qwen | 大规模具身基础模型评测 |
| Fourier | 人形策略复杂交互场景评测 |
| RoboForce | 工业策略部署前压力测试 |
| Peritas | 医疗机器人安全关键验证 |

## 对 wiki 的映射

- 与发布页合并编译 → [`wiki/entities/lightwheel-robofinals.md`](../../wiki/entities/lightwheel-robofinals.md)
- Arena 栈 → [`wiki/entities/isaac-lab-arena.md`](../../wiki/entities/isaac-lab-arena.md)
- Newton 后端 → [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)
