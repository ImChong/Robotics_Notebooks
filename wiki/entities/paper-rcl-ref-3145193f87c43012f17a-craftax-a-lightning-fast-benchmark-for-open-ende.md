---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-11-26
venue: "ICML 2024"
summary: "基于 JAX 端到端 GPU 编译的开放式强化学习基准，包含 257x 加速的 Craftax-Classic 与融合 NetHack 9 层地下城的 Craftax（169x 加速），支持 10 亿步单卡小时级实验。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_ref_3145193f87c43012f17a_craftax-a-lightning-fast-benchmark-for-o.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Craftax

**Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning**（Matthews 等，牛津大学 & UCL，ICML 2024）收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 008/564** 篇，分组 **Benchmarks & simulators**。本文构建了一套完全运行在 JAX/XLA 上的开放式生存探索强化学习基准，兼顾 NetHack 级机制深度与单卡 GPU 极速实验迭代。

## 一句话定义

基于 JAX 端到端 GPU 编译的开放式强化学习基准套件，包含高保真复刻 Crafter 的 **Craftax-Classic**（最高快 **257×**）与引入 9 层地下城、战斗属性、随机药水与附魔的 **Craftax**（最高快 **169×**），将 10 亿步 RL 训练压缩至单张 RTX 4090 约 1 小时内完成。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| JAX / XLA | Accelerated Linear Algebra | 将 Python 环境与策略训练联合编译至 GPU/TPU 消除 CPU-GPU 拷贝瓶颈 |
| UED | Unsupervised Environment Design | 无监督环境设计，通过最大化 regret 自动生成课程关卡 |
| PLR | Prioritized Level Replay | 优先关卡重放，按后悔值筛选高价值训练关卡的 UED 算法 |
| ACCEL | Adversarially Compounding Complexity by Editing Levels | 通过迭代变异关卡持续提升难度的 UED 方法 |
| E3B / RND / ICM | Elliptical Episodic Bonuses / Random Network Distillation / Intrinsic Curiosity Module | 片段级与全局级内在好奇心探索基线算法 |

## 为什么重要

- **打破「开放式复杂度」与「实验算力门槛」的二律背反**：以往开放式基准要么速度过慢（Crafter、NetHack、MineRL 需要集群才能跑大规模实验），要么机制过于简单（MiniGrid、Procgen 缺乏长期技能复用与深层探索）。
- **单卡 1 小时跑通 10 亿步开放式基准**：借助 PureJaxRL 风格的全 GPU 编译闭环与「乐观环境重置（Optimistic Environment Resets）」，研究组无需大型集群即可开展 **Craftax-1B**（长程探索/记忆/UED）与 **Craftax-1M**（秒级样本效率迭代）实验。
- **揭示当前探索与 UED 算法的能力天花板**：在 Craftax 上，PPO、PPO-RNN、RND、ICM、E3B 及 PLR/ACCEL 等主流方法在 10 亿步后最高仅拿到约 **<16%** 的理论满分，几乎无法突破第二层地下城，为长程推理与开放式学习提供了清晰试金石。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 008/564 |
| 分组 | Benchmarks & simulators |
| 出处 | ICML 2024（PMLR 235） |
| 作者 | Michael Matthews, Michael Beukman, Benjamin Ellis, Mikayel Samvelyan, Matthew Jackson, Samuel Coward, Jakob Foerster（Oxford / UCL） |
| 论文 | <https://proceedings.mlr.press/v235/matthews24a.html> |
| 代码仓库 | <https://github.com/MichaelTMatthews/Craftax>（兼容 Gymnax 与 JaxUED） |
| 子类 / 象限 | 强化学习基准 · 开放式学习与 UED · 端到端 GPU 仿真 |

## 核心机制（归纳）

### 1. 两个渐进版本：Craftax-Classic 与 Craftax

- **Craftax-Classic（Crafter 的 JAX 重写）**：保留单层 $64 \times 64$ 世界、**17** 个离散动作与 **22** 项成就（每项 +1 奖励），提供 `63×63×3` 像素观测与 **1,345 维**符号观测。为满足 JAX 静态数组形状约束，预设同时存活实体上限（僵尸 3、牛 3、骷髅 2、箭矢 3、生长植物 10），并用 Perlin 噪声替代 Simplex 噪声。
- **Craftax（融合 NetHack 机制的主基准）**：
  - **9 层程序化地下城**：地表（Floor 0）及地牢、侏儒矿洞、下水道、宝库、巨魔矿洞、火焰领域、寒冰领域、最终 Boss 层（Floor 8），层间通过梯子升降；部分楼层需放置火把照亮（光照 $< 0.05$ 时遮蔽视野块信息）。
  - **角色属性与战斗附魔**：每下潜新楼层获得经验点，可分配至敏捷、力量、智力；伤害与防御拆分为物理、火、冰三系，怪物从 3 种扩充至 **19 种**；每回合药水颜色与效果**随机重排**，逼迫智能体在回合内通过试错完成「情境学习（in-context learning）」与记忆；支持消耗魔力与宝石对武器/防具附魔。
  - **接口与奖励结构**：遵循 Gymnax 函数式状态接口，动作空间 **43 维**，最大回合步数 **100,000**；观测支持像素（`110×130×3`）、符号（**8,268 维**展平向量，含 $9\times 11$ 地图块/物品/生物 one-hot 与光照及库存开方缩放）及文本渲染器。共 **65 项成就**分为 Basic（+1）、Intermediate（+3）、Advanced（+5）、Very Advanced（+8）四档，叠加生命值增减 $\pm 0.1$，总计理论最高成就分 **226**。

### 2. JAX 编译优化与乐观环境重置（Optimistic Resets）

由于 JAX 的 `vmap` 无法执行动态分支，传统 Gymnax 写法必须在每个时间步对所有并行环境同时计算一次 `step` 和一次 `reset`（含 9 层 Perlin 噪声与房间生成），导致超过 **99%** 的重置世界被直接丢弃。Craftax 提出**乐观环境重置**：对 $N$ 个并行环境（如 4,096），每步仅生成 $M = N/16$ 个新世界，遇到 `done` 的环境从这 $M$ 个状态中无放回抽样（平均回合长约 200 步时，同时超过 $N/16$ 个环境重置的概率 $P < 10^{-10}$），从而直接获得约 **2×** 的额外端到端加速。

### 方法流程图

```mermaid
flowchart LR
    S["函数式环境状态 s_t (9层世界/属性/药水映射)"] -->|"vmap 并行 step (4096 workers)"| Obs["符号/像素/文本观测生成 (8268维或 110x130x3)"]
    Obs -->|"GPU 端策略前向 (PureJaxRL / JaxUED)"| Act["离散动作 a_t (43 维)"]
    Act -->|"XLA 编译转移与 65 项成就/血量奖励计算"| Next["新状态与 done 标志"]
    Next -->|"若 done 且按 N/16 乐观抽样"| Reset["复用预生成新世界状态 (约 2x 加速)"]
    Next -->|"否则保留 s_t+1 继续"| S
```

## 工程实践

- **基准评测协议选择**：
  - **Craftax-1B**：在 `Craftax-Symbolic` 上运行 10 亿步，单张 RTX 4090 约 **1 小时**完成，专门评测长程探索、记忆（`PPO-RNN` 使用 GRU）与持续适应；
  - **Craftax-1M**：在 `Craftax-Symbolic` 上运行 100 万步，数秒内完成，用于样本效率方法与超参的极速筛选（即使在 1M 步下，大规模并行环境仍能显著提升性能）。
- **网络骨干建议**：作者实验发现，处理符号观测时 **4 层宽度 512 的 MLP**（配合 GRU 构成 PPO-RNN）在表现上持续优于针对地图视野单独使用卷积层的结构。
- **UED 实验防坑**：Craftax 全状态暴露为单个 JAX PyTree，天然适配 JaxUED；但评测时必须在独立正常生成的 DR 关卡分布上验证，避免被 UED 回放缓冲内的分数假象误导。

## 评测与指标

### 1. 端到端训练速度对比（Table 1，单机 RTX 4090 24GB + i9-13900K）

| 环境 | 最佳并发 Workers | 端到端 Steps/s | 相对 Crafter 加速比 |
|------|------------------|----------------|---------------------|
| **Craftax-Classic (Symbolic)** | 4,096 | **405,618** | **257×** |
| **Craftax (Symbolic)** | 4,096 | **266,961** | **169×** |
| Procgen（SB3 PPO） | 1,024 | 7,638 | ~4.8× |
| NetHack / NLE（IMPALA） | 64 | 5,628 | ~3.6× |
| Crafter（SB3 PPO） | 1,024 | 1,580 | 1× |
| MineRL（随机动作多线程） | 8 | 133 | ~0.08× |

注：符号观测版本比像素观测版本快约 **10×**；在 Craftax-Classic 上，普通 PPO 运行 10 亿步可在 **51 分钟**内达到约 **90%** 理论最优回报。

### 2. Craftax-1B 探索与 UED 算法结果（10 随机种子，满分 = 226）

- **标准 RL 与内在奖励探索基线（Figure 3 & 4）**：所有方法（`PPO`、`PPO-RNN`、`ICM`、`E3B`、`RND`）的最终回报均低于满分的 **16%**（`PPO-RNN` 最高，约 14%–15%），集中掌握 Basic 与部分 Intermediate 成就；所有方法在 **Advanced** 与 **Very Advanced** 成就上几乎零突破，极少进入第 2 层（Gnomish Mines）。
- **内在奖励失效现象**：`RND`、`ICM` 均未超越 `PPO`/`PPO-RNN`，而 `E3B` 反而显著拉低总回报（尽管 `E3B` 在 `EAT_PLANT` 单项成就上表现最好，说明内在奖励虽驱动了不同探索行为，但可能干扰了密集外在奖励的主线学习）；将 `PPO-RNN` 扩训至 **100 亿步（10B）** 仍几乎无额外提升。
- **UED 基线对比（Figure 7，20 个固定正常关卡评测）**：原生 **PLR** 表现最好，其次为领域随机化（**DR**）、**PLR⊥**、**ACCEL (RSwap)** 与 **ACCEL (Noise)**，而无约束 **ACCEL (Swap)** 最差；且出现明显的**课程诱导协变量偏移（CICS）**：ACCEL (Swap) 在回放关卡上收集钻石成功率超过 **30%**，而在正常 DR 关卡上仅 **7%**。

## 与其他工作对比

- **对比 Crafter（Hafner 2021）**：Craftax-Classic 忠实复现了 Crafter 但提速 **257×**，暴露了 Crafter 原 1M 步协议外简单 PPO 即可在 51 分钟内 90% 求解的局限；主基准 Craftax 则通过多层地下城与随机药水机制将难度推高到当前算法难以逾越的水平。
- **对比 NetHack / NLE（Küttler et al. 2020）**：NLE 同样具有极深Roguelike探索需求，但 C++ CPU 仿真受限于跨设备通信（Table 1 仅 5,628 sps），Craftax 用纯 GPU 仿真实现了 ~47× 更高吞吐。
- **对比 XLand-Minigrid（Nikulin et al. 2023）与 Procgen（Cobbe et al. 2020）**：XLand-Minigrid 与 Procgen 侧重「跨大量语义各异关卡的广度泛化（wide generalisation）」，而 Craftax 聚焦「分布一致但机制深邃的深度探索（deep exploration）」。

## 结论

**Craftax 用端到端 JAX 编译把类 NetHack 深度开放式生存环境推入单卡小时级迭代时代，同时实证揭示了当前 PPO 家族、好奇心探索与 UED 方法在长程深度探索上的系统性瓶颈。**

- **科研预算受限优先选用 `Craftax-Symbolic`**：1B 步仅需单张 RTX 4090 约 1 小时，1M 步仅需数秒；只有在研究视觉表征学习时才切换至速度慢 ~10× 的像素模式。
- **务必加入循环记忆模块（如 GRU）**：`PPO-RNN` 是唯一能稳定高频进入第一层地牢并显著领先前馈 MLP 的基线，无记忆策略难以处理每回合重排药水与多层地图拓扑。
- **慎用未经调衡的内在探索奖励**：当外在成就奖励本身较密集时（分档 +1/+3/+5/+8），盲目叠加 RND/ICM/E3B 反而可能分散注意力并降低最终总分。
- **评估 UED 算法须严格区分 Replay 分数与独立测试分布**：CICS 现象会导致 ACCEL 变体在自生成回放关卡刷出虚高成绩（如采钻 30% vs 真实 7%）。
- **Craftax-Classic 仅作调试与热身**：因普通 PPO 在 1 小时内即可达到 90% 表现，新算法的核心验证应以 **Craftax-1B / Craftax-1M** 为准。

## 常见误区与局限

1. **误区：40.5 万 / 26.7 万 sps 的极速对像素输入同样成立**。Table 1 的峰值吞吐针对符号观测（`Craftax-Symbolic`）测得，开启像素渲染后速度会下降约一个数量级（~10×）。
2. **误区：在 Craftax 上做算法只需关注探索奖励或延长训练步数**。实验表明简单将 `PPO-RNN` 延长 10 倍至 100 亿步几乎没有带来额外收益，瓶颈在于长程规划、分层技能复用与策略表示，而非单纯样本量不足。
3. **JAX 静态编译带来的工程限制**：为编译固定计算图，Craftax-Classic 必须限制同屏最大生物/植物数，且乐观环境重置在平均回合极短时理论上存在轻微重置世界重复风险。
4. **领域外推边界**：Craftax 是二维离散网格符号/像素游戏，不包含刚体连续物理动力学，向真机操作或足式控制迁移时仍需结合 [Sim2Real](../concepts/sim2real.md) 与物理引擎基准。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 同组对比：[Leveraging Procedural Generation to Benchmark Reinforcement Learning（Procgen）](paper-rcl-ref-5c37def65669693c1520-leveraging-procedural-generation-to-benchmark-re.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_ref_3145193f87c43012f17a_craftax-a-lightning-fast-benchmark-for-o.md`](../../sources/papers/rcl_awesome_wam_ref_3145193f87c43012f17a_craftax-a-lightning-fast-benchmark-for-o.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://proceedings.mlr.press/v235/matthews24a.html>

## 推荐继续阅读

- [Craftax 官方开源代码库](https://github.com/MichaelTMatthews/Craftax)
- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文（ICML 2024 / PMLR 235）](https://proceedings.mlr.press/v235/matthews24a.html)
