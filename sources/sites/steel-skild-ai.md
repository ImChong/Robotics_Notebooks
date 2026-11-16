# Steel — Skild AI × NVIDIA 官方技术博客（来源归档）

- **标题：** Steel: Scaling Simulation Across Robots, Tasks, and Worlds
- **来源：** https://skild.ai/blogs/steel
- **机构：** Skild AI、英伟达（NVIDIA）
- **类型：** 官方技术博客 / 求解器项目
- **官方源码：** https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/steel
- **公告仓：** https://github.com/skild-ai/steel
- **WebGPU 移植：** https://github.com/skild-ai/steel-webgpu
- **版本 / 核验日期：** 2026-11-09 发布；2026-11-14 复核评测条件
- **核对范围：** 通读官方博客 1–5 节、5–20 类异构批次表格、单任务结果、Measurement notes（引擎版本/开关/统计范围）、Newton 集成、刚体与可变形耦合示例；核对公告仓 README 与 Newton SolverSteel 入口及阶段代码。未独立运行基准。
- **通读：** 已完成

## 可核实的主张

1. **定位：** Steel（Scalable Throughput-optimized Engine for Embodied Learning）是集成在 Newton 的 GPU 刚体动力学求解器；支持跨不同机器人、任务和世界的异构环境批处理；不是独立 VLA 训练平台。
   - 对 wiki 的映射：[Steel 实体](../../wiki/entities/steel-newton-solver.md)。
2. **主实验：** 20 个机器人学习任务中，按两引擎各自最佳测量 batch 大小的吞吐几何均值，为 PhysX 的 **3.8 倍**，其中 19/20 任务更快。
   - 对 wiki 的映射：实体「实验与评测」。
3. **异构实验：** 混合 5、10、15、20 种任务，最高为 PhysX 峰值测量吞吐的 **13.0 倍**。针对 20 种任务、61,440 环境，相同 batch 对比为 **12.0 倍吞吐与低 62% GPU 内存**；不能将两种口径混用。
   - 对 wiki 的映射：实体「评测口径与局限」。
4. **硬件及边界：** 每次运行一张 NVIDIA RTX PRO 6000 Blackwell；数字是仿真 throughput，**不是端到端策略训练速度或现实 Sim2Real 成功率**。博客的“多样性扩展”属于研究目标，而非已经测得的泛化能力提升。
   - 对 wiki 的映射：实体「局限与风险」。

## 开源核查

- **已开源：** 主求解器位于 [newton-physics/newton](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/steel)，对外接口 `newton.solvers.SolverSteel`。
- [skild-ai/steel](https://github.com/skild-ai/steel) 是公告与入口导航，真正的求解器实现位于 Newton；许可均声明 Apache-2.0。
- **另行发布：** [steel-webgpu](https://github.com/skild-ai/steel-webgpu) 提供浏览器 WebGPU 移植，并非同一 Python CUDA 求解路径。
- **源码归档：** [Newton SolverSteel](../repos/steel-newton.md)。

## 异构批次数据表（官方 Figure 3，2026-11-09）

| 混合任务类型数 | Steel 峰值吞吐（env-steps/s） | PhysX 峰值吞吐（env-steps/s） | 速度比 |
|---|---|---|---|
| 5 | 5.83M @ 61,440 | 632K @ 15,360 | 9.2× |
| 10 | 4.95M @ 122,880 | 474K @ 61,440 | 10.4× |
| 15 | 3.82M @ 122,880 | 306K @ 61,440 | 12.5× |
| 20 | 3.90M @ 245,760 | 300K @ 61,440 | 13.0× |

20 种任务在**同为 61,440 环境**时，Steel 达 3.61M、PhysX 达 300K env-steps/s（12.0×），显存分别约 11 / 29 GiB（节省约 62%）。与上表不同，这一对照控制了 batch 大小。

## 测量设置与报告边界

- **硬件：** 单张 NVIDIA RTX PRO 6000 Blackwell；测量仅为 simulation-step throughput，未覆盖观测、奖励、重置、策略训练 wall-clock 或真实机器人策略结果。
- **异构：** Newton release-1.8 candidate `01ea1293`；joint velocity limits 开启，故 `reuse_kinematics` 未激活；PhysX 基于 Isaac Sim 6.1.0.0 (omni.physx 110.3.2)；2026-11-08 至 09 测得。每环境步单次 5 ms 物理步进，CUDA Graph 开启，无策略 / domain randomization / 摄像头，动作使用微小扰动的保持位置目标。
- **单任务：** Newton 同一候选版本，但 `reuse_kinematics=True`（Newton 1.8 opt-in）；对照 MuJoCo Warp 3.17、Isaac Sim 6.1.0 PhysX；2026-11-07 测得。20 任务最佳 batch 几何均值为 PhysX 的 3.8×，MuJoCo Warp 的对应对照约为 1.3×。
- **版本敏感性：** 原文披露 Isaac Sim 6.0.0.0 的 PhysX 在这些异构批次中吞吐最多约为 6.1 的 2.7 倍；公告选用 6.1。不可把报告的最高 13× 当成不依赖版本的固定相对速度。
- **数据来源：** [Skild AI 官方技术报告 § Performance at Scale / Measurement notes](https://skild.ai/blogs/steel)。上述均是作者测得的数据，不代表本仓库已独立复验。

## 对 wiki 的增量映射

- [Steel 详情：异构批次完整数据及复测口径](../../wiki/entities/steel-newton-solver.md)
