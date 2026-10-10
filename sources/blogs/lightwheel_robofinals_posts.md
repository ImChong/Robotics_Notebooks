# 光轮 RoboFinals 官方博文合集（3 篇）

> 来源归档（blog / 厂商官方博文）

- **标题：** Lightwheel RoboFinals 官方博文三篇（发布 / 工业 benchmark 早期采用 / Newton-native 全栈）
- **类型：** blog（厂商官方博客，lightwheel.ai/blogs）
- **来源：** 光轮智能（Lightwheel）
- **博客列表：** https://lightwheel.ai/blogs（日期以列表页为准）
- **入库日期：** 2026-10-10（正文与开源状态均于 2026-10-10 重新抓取核对）
- **一句话说明：** RoboFinals 从「100 任务工业评测平台发布」→「AutoDataGen + Arena + OSMO 评测栈与早期采用方」→「首个 Newton-native 全栈 benchmark（首发 22 任务）」的三步演进；数字均为 **光轮自报**。
- **既有单篇归档：** [发布页](../sites/lightwheel_robofinals.md)、[Industrial Benchmark](../sites/lightwheel_robofinals_industrial_benchmark.md)、[Newton-native](../sites/lightwheel_robofinals_newton_native_benchmark.md)
- **沉淀到 wiki：** [`wiki/entities/lightwheel-robofinals.md`](../../wiki/entities/lightwheel-robofinals.md)

---

## 篇目清单

| 日期 | 标题 | URL |
|------|------|-----|
| 2025-12-04 | Lightwheel Unveils RoboFinals — The Industrial-Grade Simulation Evaluation Platform that Finally Challenges Frontier Robotics Foundation Models | https://lightwheel.ai/robofinals |
| 2026-03-16 | RoboFinals Industrial Benchmark — How Early Adopters Are Scaling Model Evaluation for Physical AI | https://lightwheel.ai/media/robofinals-industrial-benchmark |
| 2026-08-18 | The First Newton-Native Benchmark: Running the Full Evaluation Stack on Newton | https://lightwheel.ai/media/lightwheel-launches-robofinals-full-stack-robot-evaluation-benchmark-newton |

## 开源状态核查（2026-10-10）

| 项 | 结论 |
|----|------|
| RoboFinals 平台 / Cloud API / on-prem | **商业闭源**；发布页仍为 Book a Demo / 「Coming soon」措辞 |
| RoboFinals-100 任务包 | **未公开下载** |
| Newton-native 22 任务 + teleop 数据 | **未公开**（文内未列 GitHub / HF 链接） |
| AutoDataGen | **已开源** — Industrial Benchmark 文内直链 <https://github.com/LightwheelAI/AutoDataGen>；README 可读（包名 `autosim`，Isaac Lab + cuRobo，Python 3.12，示例 `AutoSimPipeline-FrankaCubeLift-v0`）；根目录 `LICENSE` 为 Apache 2.0。**更正**：此前归档（2026-09-06/09-14）写「无公开仓库」有误 |
| Isaac Lab-Arena | **已开源**（isaac-sim/IsaacLab-Arena） |
| LW-BenchHub | **已开源**（README 声明 Apache 2.0），见 [BenchHub/Arena 博文合集](./lightwheel_benchhub_arena_posts.md) |

## 1. 2025-12-04 · Lightwheel Unveils RoboFinals

- 定位：「业界首个足够难、工业级、可承载前沿模型」的仿真评测平台，面向 **Frontier Labs** 的 VLA；**Coming soon**。
- 痛点三条：学术仿真榜被刷满；真机评测无自动驾驶式 **shadow mode**，需「数百套物理设置」；现有仿真任务过简或设计不真实，sim–deploy 信任断层。
- **RoboFinals-100**：100 任务，基于 SimReady 资产；家庭（清洁/整理/收纳/摆放）、工厂（零件搬运/装配/机台交互）、零售（补货/分拣/货架）；刚体 / 铰接（家电、柜、冰箱、旋钮）/ 可变形（线缆、布、液体）；**统一成功判据**；三类具身（桌面臂、移动操作、全身 loco-manipulation）。示例任务：「Washing Dishes: Start Dishwasher Cycle」。
- 平台：构建于 **Isaac Lab-Arena**（文中称 "upcoming"，光轮×NVIDIA 共同开发）；完全受控、确定性的批量执行；按任务类型/难度/领域聚合指标；**Cloud API** 与 **on-premise** 两种部署。
- 多物理后端：Isaac Lab + **Newton**（主工业求解器）、Isaac Lab + PhysX、MuJoCo、Genesis → 统一记分板。
- Real2Sim / Sim2Real：SimReady 全库 Real2Sim 标定；在建受控真机 benchmark 与「业界首个」Sim–Real 相关性数据集（**尚未发布**）。
- 合作：**Qwen** 共定义部分工业场景、任务结构与评测标准，并用于高吞吐评测。

## 2. 2026-03-16 · RoboFinals Industrial Benchmark

- 口号：「Training builds capability. Evaluation defines progress.」；愿景「robotics 评测的 ImageNet」。
- **早期采用方（自报）**：Qwen（具身基础模型大规模评测）、Fourier（人形复杂交互）、RoboForce（工业策略部署前压测）、Peritas（医疗机器人安全关键验证）。
- **AutoDataGen**：基于 Isaac Lab 的自动仿真数据生成管线——LLM 从任务代码/场景配置/自然语言分解为原子技能；作为 Isaac Lab 附加包、最小侵入；统一抽象；decomposer / skill / action adapter 注册可插拔；与 **LW-BenchHub** 集成自动分解并执行 benchmark 任务；「多支团队在跑 RoboFinals 前用它生成动作数据」（自报）。
- **评测栈**：SimReady 环境 + RoboFinals-100 → Isaac Lab-Arena（environments / robots / tasks 解耦，「乐高式」）→ 光轮扩展（复杂任务逻辑、长时域工作流、通才模型评测协议）→ **NVIDIA OSMO** 编排（实验执行、调度、分布式 rollout）→ 云 GPU（含 **Nebius** 集群）→「数千 episode 并行」。

## 3. 2026-08-18 · The First Newton-Native Benchmark

- 宣称：RoboFinals（建于开源 Isaac Lab-Arena）在 **Newton** 上跑完整评测栈，并包含「首个 fully Newton-native 评测 benchmark」；**资产、求解器、机器人、teleop 数据、训练与评测** 均在 Newton 原生构建并端到端验证。
- **首发 22 任务**，三族：
  - 工厂：连接器插入对齐、从产线剔除不良料、不规则姿态拣件、线缆插接并确认端口、托盘拣选；
  - 家庭：开柜门/抽屉、冰箱取物、装洗碗机、操作微波炉门与按钮、整理台面；
  - 医院：无菌室手术器械整理、把钳/剪等铰接器械放入托盘槽与 stringer。
- 分层：资产按最新 Newton schema 建、参数来自实测标定、入库前在引擎内抓/开/插验证；Warp + Newton 刚体之上自研 **线缆/布料** 可变形求解器、多物理耦合与复杂碰撞（夹爪–可变形体无穿透）；四具身 **X7S、Dexmate、H2 plus、G1**；Newton 内原生 teleop；**每任务数百条** 示范 + 标准化质检；评测仍在 Isaac Lab-Arena，每任务带成功判据、协议与计分脚本。
- 闭环：Newton 内 teleop 采集 → Isaac Lab 训练 → RoboFinals 标准任务评测；**自述局限**：训练与评测都在仿真内，不单独证明与硬件一致；真机相关性「是下一步要发布的内容」。
- 光轮在 Newton **Technical Steering Committee**（Newton 由 NVIDIA、Google DeepMind、Disney Research 发起），负责资产标准与可变形求解器开发。

## 不可核实 / 需后续跟进

- RoboFinals-100 的 100 个任务全表、成功判据与任何模型分数：三篇文均 **未公开**。
- 22 任务的逐任务名称（文中只给任务族描述）、每任务 demo 精确条数、训练所用策略与 SR：**未披露**。
- 早期采用方的使用规模与结果：仅为光轮自述，无第三方佐证。
- Sim–Real 相关性数据集：截至 2026-10-10 未见发布。

## 对 wiki 的映射

- [`wiki/entities/lightwheel-robofinals.md`](../../wiki/entities/lightwheel-robofinals.md) — 产品层主节点（「官方博文时间线」节）
- [`wiki/entities/cn-os-lw-benchhub.md`](../../wiki/entities/cn-os-lw-benchhub.md) — 开源 BenchHub / AutoDataGen 联动
- [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md) — Newton 引擎
