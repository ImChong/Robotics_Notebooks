---
type: entity
tags:
  - entity
  - benchmark
  - vla
  - manipulation
  - loco-manipulation
  - industrial
  - isaac-lab
  - lightwheel
  - nvidia
  - simulation-evaluation
status: complete
updated: 2026-09-06
related:
  - ./isaac-lab-arena.md
  - ./lw-benchhub-tour.md
  - ./robocasa.md
  - ./dexbench.md
  - ./lerobot.md
  - ./newton-physics.md
  - ./genesis-sim.md
  - ./mujoco.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../overview/hub-embodied-eval-benchmark.md
sources:
  - ../../sources/sites/lightwheel_robofinals.md
  - ../../sources/sites/lightwheel_robofinals_industrial_benchmark.md
summary: "Lightwheel RoboFinals 是面向前沿 VLA/通才模型的工业级仿真评测平台：RoboFinals-100（100 任务、SimReady 资产）+ Isaac Lab-Arena 底座 + OSMO 编排；商业 Coming soon；底层 Arena/LW-BenchHub 已开源。"
---

# Lightwheel RoboFinals

**Lightwheel RoboFinals** 是光轮科技（Lightwheel）发布的 **工业级仿真评测平台**，面向已超越学术 benchmark 的 **VLA / 通才机器人基础模型**。核心是 **RoboFinals-100**（100 任务、SimReady 资产、跨家庭/工厂/零售），运行在 **NVIDIA Isaac Lab-Arena** 之上，并可通过 **NVIDIA OSMO** 与云 GPU 做大规模并行 rollout。平台本身为 **商业服务（Coming soon）**；开源底座为 [Isaac Lab-Arena](./isaac-lab-arena.md) 与 [LW-BenchHub](./lw-benchhub-tour.md)。

## 一句话定义

**当 LIBERO/RoboCasa 刷不动前沿 VLA 时，用 100 个工业对齐任务 + 多物理后端 + Arena 并行栈，把「评测」做成可扩展基础设施——而不是靠几百套真机硬测。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略；RoboFinals 主要评测对象 |
| SimReady | Lightwheel SimReady Asset | 光轮 Real2Sim 标定资产标准，支撑 RoboFinals-100 |
| Arena | Isaac Lab-Arena | NVIDIA×光轮联合评测框架；环境/机器人/任务解耦 |
| OSMO | NVIDIA OSMO | 分布式 AI 工作负载编排，用于大规模 benchmark rollout |
| Real2Sim | Real to Simulation | 用真机数据标定仿真资产动力学 |
| SR | Success Rate | 统一成功判据下的任务成功率 |

## 为什么重要

- **评测瓶颈叙事与工程对齐：** 光轮与 Qwen 等早期采用方共识：训练提速后，**学术仿真榜饱和 + 真机无 shadow mode** 使评测成为 Physical AI 主瓶颈（见 [具身评测选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)）。
- **与 Arena 生态锚定：** [Isaac Lab-Arena](./isaac-lab-arena.md) README 已将 RoboFinals 列为共建 benchmark；本页是 **商业任务包 + 编排服务** 的产品层，不是第二个 Arena  fork。
- **多物理后端记分板：** 同一任务可在 Isaac+Newton、Isaac+PhysX、MuJoCo、Genesis 上跑，检验 **跨仿真器鲁棒性**——与 [Newton](./newton-physics.md)、[Genesis](./genesis-sim.md) 实体直接相关。
- **工业域覆盖：** 相对 [RoboCasa](./robocasa.md) 厨房通才榜与 [DexBench](./dexbench.md) 工业灵巧**规格**，RoboFinals 强调 **长程 household + factory + retail** 与铰接/可变形交互的 **统一 SR 榜**。

## 核心结构

```mermaid
flowchart TB
  subgraph bench["RoboFinals-100"]
    T["100 任务<br/>家庭·工厂·零售"]
    A["SimReady 资产<br/>刚体·铰接·可变形"]
    E["跨具身<br/>桌面臂·移动·loco-manip"]
  end
  subgraph stack["评测栈"]
    AR["Isaac Lab-Arena"]
    LW["Lightwheel 任务/协议扩展"]
    AD["AutoDataGen<br/>合成动作数据"]
    OS["NVIDIA OSMO 编排"]
  end
  subgraph back["物理后端"]
    N["Isaac+Newton"]
    P["Isaac+PhysX"]
    M["MuJoCo"]
    G["Genesis"]
  end
  bench --> AR
  AD --> AR
  AR --> LW --> OS
  AR --> back
  back --> SC["统一记分板"]
```

| 组件 | 角色 |
|------|------|
| **RoboFinals-100** | 100 任务 benchmark；统一成功判据 |
| **SimReady** | Real2Sim 标定资产库 |
| **Isaac Lab-Arena** | 开源评测核：Scene / Embodiment / Task |
| **AutoDataGen** | LLM 分解 + Isaac Lab 包；为 benchmark 自动生成动作数据（**未开源**） |
| **OSMO + 云 GPU** | 数千 episode 并行（文内提及 Nebius 集群） |
| **Real 验证轨** | 受控真机 benchmark + Sim–Real 相关性数据集（建设中） |

## 开源与访问（步骤 2.5）

| 项 | 状态（2026-09-06） |
|----|-------------------|
| RoboFinals 平台 / API | **商业闭源**；[发布页](https://lightwheel.ai/robofinals) 标注 **Coming soon**，需 Book a Demo |
| RoboFinals-100 完整任务包 | **未公开下载**；任务域与交互类型见官方文 |
| [Isaac Lab-Arena](https://github.com/NVIDIA/IsaacLab-Arena) | **已开源** Apache 2.0 |
| [LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) | **已开源** Apache 2.0；138+ RoboCasa/LIBERO 任务 |
| AutoDataGen | 官方媒体介绍；**无公开仓库链接** |

## 早期采用方（官方文）

| 团队 | 用途 |
|------|------|
| Qwen | 共建场景与评测标准；高吞吐行业对齐评测 |
| Fourier | 人形复杂交互 |
| RoboForce | 工业策略部署前压力测试 |
| Peritas | 医疗机器人安全关键验证 |

## 工程实践

| 目标 | 做法 |
|------|------|
| 等 RoboFinals 开放前 | 先用 [Arena](./isaac-lab-arena.md) + [LW-BenchHub](./lw-benchhub-tour.md) / EnvHub 跑通评测管线 |
| 对齐工业难度 | 对照 [DexBench](./dexbench.md) OSC/Regime 语言，理解 RoboFinals 工厂域任务设计 |
| 多后端对比 | 规划同一策略在 Newton vs PhysX vs MuJoCo 的 SR 差异实验 |
| 数据飞轮 | 关注 AutoDataGen 是否开源；现阶段可参考 Tour 仓 LLM 场景扩增 + 自过滤示范 |
| 真机验证 | 跟踪光轮 Sim–Real 相关性数据集发布，勿把仿真 SR 当部署保证 |

## 局限与风险

- **不可自助复现：** 截至入库日平台需商务接入，**不能**像 LIBERO 一样 `git clone` 即跑。
- **与学术榜不可直接横比：** RoboFinals-100 难度与资产复杂度高于传统 kitchen benchmark；数值勿与 [RoboCasa](./robocasa.md) 公开榜混谈。
- **AutoDataGen 黑盒：** 合成数据管线未开源，复现「官方数据 + 官方榜」存在信息缺口。
- **多后端 ≠ 多真机：** 跨仿真器一致仍不保证真机；Real2Sim 数据集尚在建设。
- **营销叙事：** 「ImageNet of robotics」为愿景表述，需用独立第三方复现与公开榜单验证。

## 关联页面

- [Isaac Lab-Arena](./isaac-lab-arena.md) — 开源评测底座
- [LW BENCHHUB TOUR](./lw-benchhub-tour.md) — 光轮厨房 + EnvHub 工程样例
- [RoboCasa](./robocasa.md) — 厨房通才仿真榜（难度低于 RoboFinals 叙事）
- [DexBench](./dexbench.md) — 工业灵巧规格（Arena coming soon）
- [Newton Physics](./newton-physics.md) — RoboFinals 主工业求解器后端
- [具身评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)

## 参考来源

- [RoboFinals 发布页归档](../../sources/sites/lightwheel_robofinals.md)
- [Industrial Benchmark 媒体文归档](../../sources/sites/lightwheel_robofinals_industrial_benchmark.md)
- [官方发布页](https://lightwheel.ai/robofinals)
- [媒体深度文](https://lightwheel.ai/media/robofinals-industrial-benchmark)

## 推荐继续阅读

- [Isaac Lab-Arena GitHub](https://github.com/NVIDIA/IsaacLab-Arena)
- [LW-BenchHub 文档](https://docs.lightwheel.net/lw_benchhub)
- [NVIDIA 博客：通才策略评测](https://developer.nvidia.com/blog/accelerating-generalist-robot-policy-evaluation-with-nvidia-isaac-lab-arena/)
