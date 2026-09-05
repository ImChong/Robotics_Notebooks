---
type: overview
tags: [hub, embodied-eval-benchmark, benchmark, evaluation, mllm, world-model, sim2real]
status: complete
updated: 2026-09-05
related:
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../concepts/motion-control-policy-evaluation-metrics.md
  - ../concepts/sim-vs-real-eval-gap.md
  - ../entities/robo-bench.md
  - ../entities/esi-bench.md
  - ../entities/paper-daily-omni.md
  - ../entities/ewmbench.md
  - ../entities/paper-worldscore.md
  - ../entities/paper-harnesseval-w.md
  - ../entities/paper-gigaworld-1-policy-evaluation.md
  - ../entities/paper-worldecho-worldsync.md
  - ../entities/robodojo.md
  - ../entities/paper-prm-as-a-judge.md
  - ../entities/xpolicylab.md
  - ../entities/paper-softvtbench.md
  - ../entities/paper-mmhu.md
  - ../entities/dexbench.md
  - ../entities/paper-dexholdem.md
  - ../entities/paper-imitator-game.md
  - ../entities/paper-bet4sim2real.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../entities/paper-failbench.md
sources:
  - ../../sources/papers/robo_bench_arxiv_2510_17801.md
  - ../../sources/papers/ewmbench.md
  - ../../sources/papers/worldscore_arxiv_2504_00983.md
  - ../../sources/papers/harnesseval_w_arxiv_2608_16859.md
  - ../../sources/papers/esi_bench_arxiv_2605_18746.md
  - ../../sources/papers/daily_omni_arxiv_2505_17862.md
  - ../../sources/papers/robodojo_arxiv_2607_04434.md
  - ../../sources/papers/prm_as_a_judge_arxiv_2608_14284.md
  - ../../sources/papers/softvtbench_arxiv_2607_04234.md
  - ../../sources/papers/mmhu_arxiv_2507_12463.md
  - ../../sources/sites/dexbench-org.md
  - ../../sources/papers/imitator_game_arxiv_2608_22301.md
  - ../../sources/repos/the-imitator-game.md
  - ../../sources/datasets/ig-10k.md
  - ../../sources/papers/bet4sim2real_arxiv_2608_21572.md
summary: "具身评测基准选型闭环知识链枢纽：把具身大脑/MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准 四层评测，从分散的评测基准实体页收拢为一条可导航的选型链，统一各层测什么、用什么代表性基准、指标的可复现性/真实代表性/过程 vs 结果/成本取舍入口。"
---

# 具身评测基准选型闭环（知识链汇总）

> **知识链定位**：本页是「MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准」四层具身评测基准的统一入口，把近周密集 ingest 的 RoboBench / ESI-Bench / Daily-Omni / EWMBench / GigaWorld-1 等评测基准从分散的实体页收拢为一条可导航的选型链。它是「[具身大模型分类学选型闭环](./hub-embodied-foundation-model.md)」的评测姊妹篇——前者回答「选哪一类具身大模型」，本知识链回答「怎么评测/证明它」。

## 一句话定义

**具身评测基准选型闭环** 指按 **具身大脑/MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准** 逐层分工的评测谱系，各层共享「测什么 / 用什么代表性基准 / 指标怎么读」的方法学底座，但在可复现性、真实代表性、过程 vs 结果指标、成本上各有取舍，需按评测目的组合选型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MLLM | Multimodal Large Language Model | 多模态大模型，作为具身大脑的认知评测对象 |
| EWM | Embodied World Model | 具身世界模型，视频生成保真度评测对象 |
| WM | World Model | 世界模型，时序预演与预测保真度 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略，成功率评测对象 |
| WMES | World Model Evaluation Score | GigaWorld-1 世界模型评估综合分 |
| gap | Sim-to-Real Evaluation Gap | 仿真评测结论外推真机的偏差 |

## 为什么重要

- **补一条贯通的评测选型视角**：仓库已有各评测基准的实体页，但缺「从认知到真机逐层测什么、各基准边界与取舍」的统一决策入口。
- **暴露评测层间取舍矛盾**：仿真基准易复现 vs 真机代表性、任务成功率 vs 过程/中间指标、世界模型视频质量 ≠ 下游策略收益、MLLM 认知评分 ≠ 可执行动作能力——这些矛盾只有并置在一条链上才看得清（详见事实库对应矛盾检测规则）。
- **与选型闭环同向**：选出一类具身大模型后，唯有可信评测才能证明其收益，评测选型是模型选型的验收环节。

## 四层评测选型闭环

| 层次 | 测什么 | 代表基准 | 站内入口 |
|------|--------|----------|----------|
| ① 认知评测 | MLLM 作为 embodied brain 的感知/规划/推理能力；另含日常 AV 时序对齐；驾驶人本 Behavior VQA 见 MMHU | RoboBench、ESI-Bench、Daily-Omni；驾驶相邻 **MMHU** | [RoboBench](../entities/robo-bench.md)、[ESI-Bench](../entities/esi-bench.md)、[Daily-Omni](../entities/paper-daily-omni.md)、[MMHU](../entities/paper-mmhu.md) |
| ② 预测保真度评测 | 世界模型视频生成的时序/轨迹/语义保真度；开放域多场景世界生成另见 WorldScore；交互干预/持久另见 HarnessEval-W；off-expert 动作跟随另见 WorldEcho | EWMBench、GigaWorld-1 WMBench；WorldScore / HarnessEval-W（相邻）；WorldEcho | [EWMBench](../entities/ewmbench.md)、[GigaWorld-1 策略评估](../entities/paper-gigaworld-1-policy-evaluation.md)、[WorldScore](../entities/paper-worldscore.md)、[HarnessEval-W](../entities/paper-harnesseval-w.md)、[WorldEcho / WorldSync](../entities/paper-worldecho-worldsync.md) |
| ③ 策略成功率评测 | 下游 VLA/策略的任务成功率与泛化 | GigaWorld-1 评估器、仿真闭环、**RoboDojo**；接触安全另见 **SoftVTBench**；工业灵巧规格见 **DexBench**（评测仓待发布）；真机扑克灵巧见 **DexHoldem**（报 SPSR）；成功判据本身另见 **Imitator Game**（目标等价而非轨迹相似） | [GigaWorld-1 策略评估](../entities/paper-gigaworld-1-policy-evaluation.md)、[RoboDojo](../entities/robodojo.md)、[SoftVTBench](../entities/paper-softvtbench.md)、[DexBench](../entities/dexbench.md)、[DexHoldem](../entities/paper-dexholdem.md)、[Imitator Game](../entities/paper-imitator-game.md)、[仿真评测基建](../concepts/simulation-evaluation-infrastructure.md) |
| ④ sim↔real gap 校准 | 评测结论能否外推到真机 | real-to-sim 相关性、RoboDojo RealEval、代表性代价；真机样本量不足另见 **Bet4Sim2Real**（仿真库下注换 anytime-valid 区间） | [仿真 vs 真机评测 gap](../concepts/sim-vs-real-eval-gap.md)、[RoboDojo](../entities/robodojo.md)、[Bet4Sim2Real](../entities/paper-bet4sim2real.md) |
| ③′ 运控横切 | 被测对象换成 locomotion / whole-body tracking / MPC-WBC 时的指标体系：跟踪误差、命令跟随、求解实时性、硬件裕度 | HumanTracker、HumanoidBench、TrackerLab、Barkour | [运控模型评测指标](../concepts/motion-control-policy-evaluation-metrics.md)、[HumanTracker](../entities/paper-humantracker.md)、[HumanoidBench](../entities/humanoid-bench.md)、[TrackerLab](../entities/trackerlab.md) |
| 端到端 | 四层如何逐层选型取舍 | 选型决策树 | [评测基准选型闭环 Query](../queries/embodied-eval-benchmark-selection-loop.md) |

## 评测选型的关键取舍

- **可复现性 vs 真实代表性**：仿真基准在吞吐/可控/可复现上占优，代价是牺牲真实接触、感知噪声与长尾分布的代表性；评测结论能否外推真机取决于 real-to-sim 相关性。
- **过程指标 vs 结果指标**：任务成功率（结果）直观但掩盖长尾失败模式；过程/中间指标可归因但可能与真实收益脱钩。
- **代理指标 ≠ 下游收益**：世界模型视频质量高 ≠ 下游策略收益高、MLLM 认知评分高 ≠ 可执行动作能力强，跨层用代理指标要警惕。
- **单任务过拟合 vs 跨任务泛化**：基准饱和 ≠ 真实场景就绪，评测集泄漏与分布漂移会致虚高。

## 与其他知识链的关系

- **[具身大模型分类学选型闭环](./hub-embodied-foundation-model.md)**：模型选型的家族谱系，本知识链为其验收环节。
- **[仿真到现实（Sim2Real）](./hub-sim2real.md)**：④ 层 sim↔real gap 校准与 sim2real 迁移共享同一物理根因。

## 关联页面

- [具身大模型评测基准选型闭环 Query](../queries/embodied-eval-benchmark-selection-loop.md)
- [FailBench](../entities/paper-failbench.md) — VLM 失败裁判跨源基准；接触装配 balanced accuracy ≤0.60（③ 层相邻）
- [运控模型评测指标](../concepts/motion-control-policy-evaluation-metrics.md) — 运控模型（locomotion / WBT / MPC-WBC）侧的指标体系，与 ③ 层任务成功率互补
- [仿真 vs 真机评测 gap](../concepts/sim-vs-real-eval-gap.md)
- [RoboBench](../entities/robo-bench.md)
- [ESI-Bench](../entities/esi-bench.md)
- [Daily-Omni](../entities/paper-daily-omni.md) — 日常音视频跨模态时序对齐（① 层）
- [EWMBench](../entities/ewmbench.md)
- [WorldScore](../entities/paper-worldscore.md) — 开放域 3D/4D/视频多场景世界生成统一榜（② 层相邻，非操纵轴）
- [HarnessEval-W](../entities/paper-harnesseval-w.md) — 交互式世界模型 agentic 评测（干预/持久证据树；② 层相邻，非操纵轴）
- [WorldEcho / WorldSync](../entities/paper-worldecho-worldsync.md) — off-expert 动作跟随（视觉门控 + \(\mathrm{SE}(3)\) NDTW；② 层）
- [GigaWorld-1 策略评估](../entities/paper-gigaworld-1-policy-evaluation.md)
- [RoboDojo](../entities/robodojo.md)
- [PRM-as-a-Judge](../entities/paper-prm-as-a-judge.md) — 过程评测 OPD（③ 层，可挂已有 rollout）
- [SoftVTBench](../entities/paper-softvtbench.md) — 可变形视触觉 Goal/Safety Success（③ 层过程安全）
- [DexBench](../entities/dexbench.md) — 工业灵巧规格（OSC / 18 任务）；规范已公开，Arena 仍标 coming soon
- [DexHoldem](../entities/paper-dexholdem.md) — ③ 层真机扑克灵巧：SPSR ≠ TCR；策略仓与数据已开源
- [Imitator Game](../entities/paper-imitator-game.md) — L0–L3 意图级模仿基准（③ 层成功判据：目标等价而非轨迹相似）
- [Bet4Sim2Real](../entities/paper-bet4sim2real.md) — 仿真库下注换 anytime-valid 真机性能证书（④ 层，真机样本贵时收窄区间）
- [MMHU](../entities/paper-mmhu.md) — 驾驶场景人体行为多模态基准（① 层相邻）
- [XPolicyLab](../entities/xpolicylab.md)
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)

## 参考来源

- [RoboBench 论文](../../sources/papers/robo_bench_arxiv_2510_17801.md) — MLLM 具身大脑五维评测
- [EWMBench 论文](../../sources/papers/ewmbench.md) — 具身世界模型视频生成评测
- [WorldScore 论文](../../sources/papers/worldscore_arxiv_2504_00983.md) — 多场景相机可控世界生成统一评测
- [HarnessEval-W 论文](../../sources/papers/harnesseval_w_arxiv_2608_16859.md) — 交互式世界模型 agentic 评测
- [ESI-Bench 论文](../../sources/papers/esi_bench_arxiv_2605_18746.md) — 具身空间智能评测
- [Daily-Omni 论文](../../sources/papers/daily_omni_arxiv_2505_17862.md) — 日常 AV 跨模态时序对齐
- [RoboDojo 论文](../../sources/papers/robodojo_arxiv_2607_04434.md) — 统一 sim-and-real 通用操纵评测
- [PRM-as-a-Judge 论文](../../sources/papers/prm_as_a_judge_arxiv_2608_14284.md) — 过程评测 OPD
- [SoftVTBench 论文](../../sources/papers/softvtbench_arxiv_2607_04234.md) — 可变形视触觉 Goal/Safety Success
- [MMHU 论文](../../sources/papers/mmhu_arxiv_2507_12463.md) — 驾驶人本 Behavior VQA / 运动评测
- [Imitator Game 论文](../../sources/papers/imitator_game_arxiv_2608_22301.md) — L0–L3 意图级模仿评测协议
- [Bet4Sim2Real 论文](../../sources/papers/bet4sim2real_arxiv_2608_21572.md) — 下注式 anytime-valid 真机性能证书
- 本页归纳自 [评测基准选型闭环 Query](../queries/embodied-eval-benchmark-selection-loop.md) 及各评测基准实体/概念页
