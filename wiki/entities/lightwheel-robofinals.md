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
updated: 2026-10-10
related:
  - ./lightwheel.md
  - ./cn-os-lw-benchhub.md
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
  - ../../sources/blogs/lightwheel_robofinals_posts.md
  - ../../sources/blogs/lightwheel_benchhub_arena_posts.md
  - ../../sources/sites/lightwheel_robofinals.md
  - ../../sources/sites/lightwheel_robofinals_newton_native_benchmark.md
  - ../../sources/sites/lightwheel_robofinals_isaac_lab_arena.md
  - ../../sources/sites/lightwheel_robofinals_industrial_benchmark.md
  - ../../sources/repos/isaaclab_arena.md
summary: "Lightwheel RoboFinals 是面向前沿 VLA/通才模型的工业级仿真评测平台：2025-12-04 发布 RoboFinals-100（100 任务），2026-03-16 公布 AutoDataGen+Arena+OSMO 评测栈与早期采用方，2026-08-18 推出首个 Newton-native 全栈 benchmark（首发 22 家庭/医院/工厂任务）；平台商业闭源，Arena/LW-BenchHub/AutoDataGen/Newton 已开源。"
institutions:
  - lightwheel
---

# Lightwheel RoboFinals

**Lightwheel RoboFinals** 是光轮智能（Lightwheel）发布的 **工业级仿真评测平台**，面向已超越学术 benchmark 的 **VLA / 通才机器人基础模型**。**2025-12-04** 发布，长期目标是 **RoboFinals-100**（100 任务、SimReady 资产、跨家庭/工厂/零售）；**2026-08-18** 发布自称业界首个 **Newton-native 全栈评测 benchmark**——资产、求解器、机器人、遥操作数据、训练与评测均在 [Newton](./newton-physics.md) 上原生构建并端到端验证，首发 **22** 个家庭/医院/工厂接触丰富任务。评测执行层仍经 **NVIDIA Isaac Lab-Arena** + **BenchHub**（开源实现见 [LW-BenchHub](./cn-os-lw-benchhub.md)），并可通过 **NVIDIA OSMO** 与云 GPU 大规模并行 rollout。平台为 **商业服务（Coming soon）**；开源底座为 [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena)、LW-BenchHub 与 AutoDataGen。公司背景见 [光轮智能](./lightwheel.md)。

## 一句话定义

**当 LIBERO/RoboCasa 刷不动前沿 VLA 时，用 100 个工业对齐任务 + 多物理后端 + Arena 并行栈，把「评测」做成可扩展基础设施——而不是靠几百套真机硬测。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略；RoboFinals 主要评测对象 |
| SimReady | Lightwheel SimReady Asset | 光轮 Real2Sim 标定资产标准，支撑 RoboFinals-100 |
| Arena | Isaac Lab-Arena | NVIDIA×光轮联合评测框架；环境/机器人/任务解耦 |
| OSMO | NVIDIA OSMO | 分布式 AI 工作负载编排，用于大规模 benchmark rollout |
| TSC | Technical Steering Committee | Newton 技术指导委员会；光轮参与并主导可变形求解器 |
| BenchHub | Lightwheel BenchHub | Arena 之上的 benchmark 托管/执行/规模化层（实现见 LW-BenchHub） |
| Real2Sim | Real to Simulation | 用真机数据标定仿真资产动力学 |
| SR | Success Rate | 统一成功判据下的任务成功率 |

## 为什么重要

- **评测瓶颈叙事与工程对齐：** 光轮与 Qwen 等早期采用方共识：训练提速后，**学术仿真榜饱和 + 真机无 shadow mode** 使评测成为 Physical AI 主瓶颈（见 [具身评测选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)）。
- **与 Arena 生态锚定：** [Isaac Lab-Arena](./isaac-lab-arena.md) README 已将 RoboFinals 列为共建 benchmark；本页是 **商业任务包 + 编排服务** 的产品层，不是第二个 Arena  fork。
- **多物理后端记分板：** 同一任务可在 Isaac+Newton、Isaac+PhysX、MuJoCo、Genesis 上跑，检验 **跨仿真器鲁棒性**——与 [Newton](./newton-physics.md)、[Genesis](./genesis-sim.md) 实体直接相关。
- **Newton 从后端到全栈：** 2026 媒体文称 RoboFinals 为 **首个 fully Newton-native benchmark**——不仅是换求解器，而是 SimReady 资产按 Newton schema 重建、线缆/布料可变形求解器、四类具身（X7S / Dexmate / H2+ / G1）与 **Newton 内 teleop 数据** 全链路验证；光轮在 Newton **TSC** 制定资产标准。
- **工业域覆盖：** 相对 [RoboCasa](./robocasa.md) 厨房通才榜与 [DexBench](./dexbench.md) 工业灵巧**规格**，RoboFinals 强调 **长程 household + factory + retail** 与铰接/可变形交互的 **统一 SR 榜**。

## 官方博文时间线

> 三篇官方博文的逐篇要点；原文与数字归档见 [RoboFinals 博文合集](../../sources/blogs/lightwheel_robofinals_posts.md)。所有任务数、采用方与效果均为 **光轮自报**，截至 2026-10-10 无第三方公开复现。底座层（Arena / BenchHub）的三篇技术文见 [LW-BenchHub](./cn-os-lw-benchhub.md)。

### 2025-12-04 · Lightwheel Unveils RoboFinals（发布）

- **问题陈述：** 前沿 VLA 已刷满学术仿真榜；真机评测没有自动驾驶式 **shadow mode**，需要「数百套物理设置」+ 持续维护 + 安全流程；已有仿真任务过简或设计失真，造成 sim–deploy 信任断层。
- **RoboFinals-100：** 100 任务、渐进难度；家庭 / 工厂 / 零售三域；刚体 + 铰接 + 可变形（线缆、布、液体）；统一成功判据；桌面臂 / 移动操作 / loco-manipulation 三类具身。
- **平台：** 建于（当时称 "upcoming" 的）Isaac Lab-Arena；受控、确定性批量执行，按任务类型/难度/领域聚合指标；**Cloud API** 与 **on-premise** 两种部署；Newton / PhysX / MuJoCo / Genesis 多后端统一记分板。
- **Real2Sim 双轨：** SimReady 全库 Real2Sim 标定 + 在建受控真机 benchmark 与 Sim–Real 相关性数据集（**截至 2026-10-10 未发布**）。
- **合作：** Qwen 共定义部分工业场景、任务结构与评测标准。

### 2026-03-16 · RoboFinals Industrial Benchmark（早期采用与评测栈）

- **采用方扩展到四家（自报）：** Qwen、Fourier、RoboForce、Peritas（医疗，安全关键验证），见下文「早期采用方」表。
- **AutoDataGen 首次公开亮相：** 基于 Isaac Lab 的自动动作数据生成（LLM 分解 → 原子技能 → cuRobo 规划/导航执行）；文内直链 [LightwheelAI/AutoDataGen](https://github.com/LightwheelAI/AutoDataGen)，**已开源（Apache 2.0，2026-10-10 核查）**；与 LW-BenchHub 联动（仓内 9 个示例 pipeline）。
- **评测栈成型：** SimReady + RoboFinals-100 → Arena（环境/机器人/任务解耦）→ 光轮扩展（复杂任务逻辑、长时域、通才评测协议）→ **NVIDIA OSMO** 编排 → 云 GPU（含 **Nebius**）→「数千 episode 并行」。
- **吞吐依据：** 同期（2026-02-04）光轮 × NVIDIA 性能研究自报 Arena GPU 并行较 MuJoCo/RoboCasa 顺序最高 **13.5×**（4,096 env，8× RTX 6000D，GR00T N1.5），细节见 [LW-BenchHub 页](./cn-os-lw-benchhub.md)。
- **愿景：**「robotics 评测的 ImageNet」——属定位表述，非已达成事实。

### 2026-08-18 · The First Newton-Native Benchmark（Newton 全栈）

- **从后端到全栈：** 不是把旧任务指向新引擎，而是资产（Newton schema + 实测标定 + 引擎内交互验证）、求解器（Warp/Newton 刚体 + 自研线缆/布料可变形与多物理耦合）、机器人（X7S / Dexmate / H2+ / G1）、Newton 内 teleop、每任务数百条质检示范、评测协议与计分脚本 **逐层重建**。
- **首发 22 任务：** 工厂（连接器插入、剔除不良料、不规则姿态拣件、线缆插接确认端口、托盘拣选）/ 家庭（柜门抽屉、冰箱取物、装洗碗机、微波炉门与按钮、整理台面）/ 医院（无菌室器械整理、钳/剪入托盘槽与 stringer）；只公布任务族，**未公布逐任务清单与任何策略分数**。
- **闭环与自述局限：** teleop 采集 → Isaac Lab 训练 → RoboFinals 评测在 Newton 上跑通，证明「栈内一致、可复跑」；但训练与评测都在仿真内，**不证明真机一致**，真机相关性「下一步发布」。
- **治理位置：** 光轮在 Newton TSC，负责资产标准与可变形求解器（见 [Newton Physics](./newton-physics.md)）。

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
  subgraph newton["Newton-native 全栈（首发 22 任务）"]
    AS["SimReady 资产<br/>实测标定"]
    SOL["Warp+Newton<br/>+可变形/耦合"]
    ROB["X7S·Dexmate·H2+·G1"]
    TEL["Newton 内 teleop"]
    DAT["数百 demo/任务"]
  end
  subgraph back["多物理后端记分板"]
    N["Isaac+Newton"]
    P["Isaac+PhysX"]
    M["MuJoCo"]
    G["Genesis"]
  end
  newton --> AR
  bench --> AR
  AD --> AR
  AR --> LW --> OS
  AR --> back
  back --> SC["统一记分板"]
```

| 组件 | 角色 |
|------|------|
| **RoboFinals-100** | 100 任务 benchmark 路线图；统一成功判据 |
| **Newton-native 首发集** | **22 任务**（家庭/医院/工厂）；全栈 Newton 构建；接触丰富长时域 |
| **BenchHub** | Arena 之上 benchmark 托管/执行；域随机化 + teleop 校准 horizon/成功判据 |
| **SimReady** | Real2Sim 标定资产库 |
| **Isaac Lab-Arena** | 开源评测核：Scene / Embodiment / Task |
| **AutoDataGen** | LLM 分解 + Isaac Lab 包；为 benchmark 自动生成动作数据（[已开源](https://github.com/LightwheelAI/AutoDataGen)，Apache 2.0） |
| **OSMO + 云 GPU** | 数千 episode 并行（文内提及 Nebius 集群） |
| **Real 验证轨** | 受控真机 benchmark + Sim–Real 相关性数据集（建设中） |

## 开源与访问（步骤 2.5）

| 项 | 状态（2026-10-10 复核） |
|----|-------------------|
| RoboFinals 平台 / API | **商业闭源**；[发布页](https://lightwheel.ai/robofinals) 标注 **Coming soon**，需 Book a Demo |
| RoboFinals-100 完整任务包 | **未公开下载**；任务域与交互类型见官方文 |
| **Newton-native 22 任务** | **平台内**；全栈资产/数据/协议未公开；媒体文列任务族与闭环管线 |
| Newton teleop 数据集 | 文称每任务数百条 + 质检；**未列公开 HF/GitHub** |
| [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) | **已开源** Apache 2.0 |
| [Newton](https://github.com/newton-physics/newton) | **已开源** Apache 2.0（Linux Foundation） |
| [LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) | **已开源**（README 声明 Apache 2.0，根目录无独立 LICENSE 文件）；268 任务（LIBERO 130 + RoboCasa 138），详见 [LW-BenchHub](./cn-os-lw-benchhub.md) |
| [AutoDataGen](https://github.com/LightwheelAI/AutoDataGen) | **已开源** Apache 2.0（2026-03-16 博文直链；此前「无公开仓库」记载有误，已更正） |

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
| 等 RoboFinals 开放前 | 先用 [Arena](./isaac-lab-arena.md) + [LW-BenchHub](./cn-os-lw-benchhub.md) / EnvHub 跑通评测管线（SmolVLA 样例见 [Tour](./lw-benchhub-tour.md)） |
| 对齐工业难度 | 对照 [DexBench](./dexbench.md) OSC/Regime 语言，理解 RoboFinals 工厂域任务设计 |
| 多后端对比 | 规划同一策略在 Newton vs PhysX vs MuJoCo 的 SR 差异实验 |
| Newton 全栈 | 区分「仅换后端」与 **Newton-native 资产+teleop+评测**；首发 22 任务覆盖工厂线缆插拔、医院器械整理等 |
| 数据飞轮 | AutoDataGen 已开源：可在 LW-BenchHub 内跑 `scripts/autosim/run_autosim_example.py` 示例 pipeline；注意脚本化技能序列「跑完」≠ 任务成功（Tour 仓反例） |
| 真机验证 | 跟踪光轮 Sim–Real 相关性数据集发布，勿把仿真 SR 当部署保证 |

## 局限与风险

- **不可自助复现：** 截至入库日平台需商务接入，**不能**像 LIBERO 一样 `git clone` 即跑。
- **与学术榜不可直接横比：** RoboFinals-100 难度与资产复杂度高于传统 kitchen benchmark；数值勿与 [RoboCasa](./robocasa.md) 公开榜混谈。
- **官方数据仍是黑盒：** AutoDataGen 管线已开源，但 RoboFinals 官方任务包、Newton teleop 数据与榜单数据未公开，复现「官方数据 + 官方榜」仍有缺口。
- **多后端 ≠ 多真机：** 跨仿真器一致仍不保证真机；Real2Sim 数据集尚在建设。
- **Newton 闭环自述：** 官方文承认训练/评测均在仿真内，**不单独证明真机一致**；Sim–Real 相关性结果计划后续发布。
- **营销叙事：** 「ImageNet of robotics」为愿景表述，需用独立第三方复现与公开榜单验证。

## 关联页面

- [光轮智能](./lightwheel.md) — 公司主页（SimReady / EgoSuite / RoboFinals 产品线）
- [LW-BenchHub](./cn-os-lw-benchhub.md) — BenchHub 开源实现、Arena 迁移与 13.5× 并行评测研究
- [Isaac Lab-Arena](./isaac-lab-arena.md) — 开源评测底座
- [LW BENCHHUB TOUR](./lw-benchhub-tour.md) — 光轮厨房 + EnvHub 工程样例
- [RoboCasa](./robocasa.md) — 厨房通才仿真榜（难度低于 RoboFinals 叙事）
- [DexBench](./dexbench.md) — 工业灵巧规格（Arena coming soon）
- [Newton Physics](./newton-physics.md) — RoboFinals 主工业求解器后端
- [具身评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)

## 参考来源

- [RoboFinals 官方博文合集（2025-12-04 / 2026-03-16 / 2026-08-18）](../../sources/blogs/lightwheel_robofinals_posts.md)
- [BenchHub × Arena 官方博文合集](../../sources/blogs/lightwheel_benchhub_arena_posts.md)
- [RoboFinals 发布页归档](../../sources/sites/lightwheel_robofinals.md)
- [Newton-native Benchmark 媒体文归档](../../sources/sites/lightwheel_robofinals_newton_native_benchmark.md)
- [Isaac Lab-Arena / BenchHub 技术文归档](../../sources/sites/lightwheel_robofinals_isaac_lab_arena.md)
- [Industrial Benchmark 媒体文归档](../../sources/sites/lightwheel_robofinals_industrial_benchmark.md)
- [Isaac Lab-Arena 仓库归档](../../sources/repos/isaaclab_arena.md)
- [官方发布页](https://lightwheel.ai/robofinals)
- [Newton-native 媒体文](https://lightwheel.ai/media/lightwheel-launches-robofinals-full-stack-robot-evaluation-benchmark-newton)
- [Industrial Benchmark 媒体文](https://lightwheel.ai/media/robofinals-industrial-benchmark)
- [AutoDataGen 仓库](https://github.com/LightwheelAI/AutoDataGen)

## 推荐继续阅读

- [Isaac Lab-Arena GitHub](https://github.com/isaac-sim/IsaacLab-Arena)
- [Newton-native 全栈评测文](https://lightwheel.ai/media/lightwheel-launches-robofinals-full-stack-robot-evaluation-benchmark-newton)
- [LW-BenchHub 文档](https://docs.lightwheel.net/lw_benchhub)
- [NVIDIA 博客：通才策略评测](https://developer.nvidia.com/blog/accelerating-generalist-robot-policy-evaluation-with-nvidia-isaac-lab-arena/)
