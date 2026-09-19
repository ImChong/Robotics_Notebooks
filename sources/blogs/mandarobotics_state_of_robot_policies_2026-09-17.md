# Understanding the Limits of Open-Source General Robotics Policies（Manda Robotics, 2026-09-17）

> 来源归档（blog / 第三方独立评测报告）

- **标题：** Understanding the Limits of Open-Source General Robotics Policies
- **副标题：** State of General Robot Policies 2026
- **类型：** blog / independent evaluation study
- **作者/机构：** [Manda Robotics](https://mandarobotics.com/)
- **发表日期：** 2026-09-17
- **原始链接：** <https://mandarobotics.com/blog/state-of-robot-policies/index.html>
- **入库日期：** 2026-09-19
- **一句话说明：** Manda 在 **RoboLab-120 × DROID** 上对 **5 个开源通用操纵策略** 做 **6,000 episode** 零样本 head-to-head：Cosmos 3 Nano Policy **35.1% SR** 领先但 **829 ms/step**；π0.5 **27.5% / 128 ms** 性价比较稳；**无策略可 drop-in 部署**；强调阶段化失败模式与评测粒度。

## 核心摘录

### 1) 研究问题与设置

| 项 | 内容 |
|----|------|
| **RQ1** | 开源策略零样本能力如何？ |
| **RQ2** | 共有/特有失败模式？ |
| **RQ3** | 对训练与评测的启示？ |
| **规模** | 5 policies × 120 tasks × 10 episodes = **6,000** episodes |
| **仿真** | Isaac Sim **6** + Isaac Lab **3**（[ymetz/RoboLab](https://github.com/ymetz/RoboLab) Isaac Sim 6 port） |
| **本体** | RoboLab 内置 **DROID** 固定基座单臂平行夹爪 |
| **协议** | 各策略 **native adapter**；推理 wall-clock **不计入** sim 任务预算 |

### 2) 五策略 aggregate（canonical run）

| 策略 | Success rate | Score | 推理延迟（均值） |
|------|-------------|-------|------------------|
| **Cosmos 3 Nano Policy** | **35.1%** | 50.7 | 829 ms/step |
| **π0.5-DROID (joint pos)** | 27.5% | 42.8 | **128 ms/step** |
| **MolmoAct 2** | 13.8% | — | — |
| **GR00T N1.7** | 10.2% | — | — |
| **G0.5** | 10.5% | — | — |

- **Retrospective oracle**（每 episode 选最优策略）：**49.4%** SR
- **30/120** 任务上 **五策略全失败**
- 与官方 RoboLab leaderboard 对齐：π0.5 27.5% vs 28.0%；Cosmos 35.1% vs 36.8%

### 3) 能力切片（无单一策略通吃）

| 能力 | 最强 | 最弱 |
|------|------|------|
| Target selection | MolmoAct 2 | GR00T N1.7 |
| Getting a grip | Cosmos, π0.5 | GR00T |
| Approach angle | Cosmos | MolmoAct 2 |
| Transport & release | Cosmos | G0.5 |
| Search | Cosmos, π0.5 | GR00T |
| Endurance | Cosmos | GR00T |

- π0.5 vs Cosmos **success union 仅 38.3% 重叠** — 相似总分 ≠ 相似行为
- Counting 任务：π0.5 **68.6%** vs Cosmos **54.3%**（aggregate 顺序可反转）

### 4) 策略特有失败模式（视频 + 1,150 集人工复核）

| 策略 | 典型失败 |
|------|----------|
| **π0.5** | 抓对但 **容器 rim 碰撞**；lift height 不区分成败 |
| **Cosmos** | **抖动**（jitter index 12.07 vs 4.05–8.27）；** opportunistic 放弃最后 1 cm** |
| **MolmoAct 2** | **腕角不变**反复尝试；clutter 后无法换角度 |
| **GR00T N1.7** | 抓 **画面中心** 错物体；episode 末段 **coherence 崩溃** |
| **G0.5** | **test grip 循环**（碰一下、抬 1 cm、松开）；54% episode 才真正闭合夹爪 |

### 5) 评测方法论警示

- **π0.5 同 seed 双跑：** 同 outcome 仅 **64%**；28/120 任务 SR 波动 ≥20 pp — **行为比 SR 更可复现**
- **SPARC vs 视觉抖动：** Cosmos SPARC 最平滑，但 velocity jitter 最高 — **单运动指标不足**
- **Benchmark  contingent ranking：** RoboDojo 上 G0.5 > π0.5，RoboLab 上相反 — 需 **跨 suite 三角测量**
- **开发–评测反馈环：** Cosmos 开发期用过 RoboLab；RoboLab 作者开发期用过 π0.5 — **非故意 overfit 也可能产生兼容偏置**

### 6) 对训练/评测的四条建议

1. 细粒度行为指标：acquisition / transport / release / recovery / terminal stability
2. 独立设计 benchmark + hold-out + 重复 run
3. 多 embodiment（双手、灵巧手、人形）+ latency budget
4. 检查 policy 预测序列 vs 执行轨迹

## 开源核查（步骤 2.5，2026-09-19）

| 组件 | 状态 |
|------|------|
| **评测报告** | 公开博客 + 交互图表/视频（非论文） |
| **RoboLab fork** | **已开源** — [ymetz/RoboLab](https://github.com/ymetz/RoboLab)（Isaac Sim 6 port + `docs/isaac_sim_6.md`） |
| **被测策略 checkpoint** | 各厂商 Hugging Face / GCS 分发（见文内 References） |
| **Manda 训练代码** | **不适用** — 第三方评测，非策略训练方 |

## 对 wiki 的映射

- 实体页：[`wiki/entities/manda-robotics-open-policy-evaluation.md`](../../wiki/entities/manda-robotics-open-policy-evaluation.md)
- 站点：[`sources/sites/manda-robotics-state-of-policies.md`](../sites/manda-robotics-state-of-policies.md)
- 机构：[`sources/sites/manda-robotics.md`](../sites/manda-robotics.md)
- Fork：[`sources/repos/robolab-ymetz-isaac-sim6.md`](../repos/robolab-ymetz-isaac-sim6.md)
- 交叉：[RoboLab 实体](../../wiki/entities/robolab.md)、[具身评测选型 hub](../../wiki/overview/hub-embodied-eval-benchmark.md)、[π0.5](../../wiki/entities/paper-pi05-open-world-vla.md)
