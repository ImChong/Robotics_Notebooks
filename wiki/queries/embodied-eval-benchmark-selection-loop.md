---
type: query
tags: [benchmark, evaluation, embodied-ai, mllm, world-model, vla, sim2real, taxonomy]
status: complete
updated: 2026-07-23
summary: "具身大模型评测基准选型闭环知识链：把具身大脑/MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准 四层评测，从分散的评测基准实体页沉淀为一条端到端选型决策链，逐层说明测什么、用什么代表性基准、指标的可复现性/真实代表性/过程 vs 结果/成本如何取舍及典型误判。"
sources:
  - ../../sources/papers/robo_bench_arxiv_2510_17801.md
  - ../../sources/papers/ewmbench.md
  - ../../sources/papers/esi_bench_arxiv_2605_18746.md
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
  - ../../sources/papers/driftworld_arxiv_2607_15065.md
  - ../../sources/papers/ctrl_world_arxiv_2510_10125.md
related:
  - ../overview/topic-embodied-eval-benchmark.md
  - ../concepts/sim-vs-real-eval-gap.md
  - ../entities/robo-bench.md
  - ../entities/esi-bench.md
  - ../entities/ewmbench.md
  - ../entities/paper-gigaworld-1-policy-evaluation.md
  - ../entities/paper-driftworld.md
  - ../entities/paper-masked-visual-actions.md
  - ../entities/paper-ctrl-world.md
  - ../entities/vla-sota-leaderboard.md
  - ../entities/paper-fabrivla.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../concepts/sim2real.md
  - ../queries/embodied-fm-taxonomy-loop.md
---

# Query：具身大模型评测基准选型闭环知识链

> **Query 产物**：本页由以下问题触发：「我训了一个具身大模型，接下来到底怎么『测/证明它』——从『大脑看懂没』到『真机能不能做成』中间分几层评测、每层测什么、用哪个代表性基准、指标要复现性还是真实代表性、看过程指标还是结果指标、哪一层评测结论最容易骗人？」
> 综合来源：[RoboBench（MLLM 具身大脑评测）](../entities/robo-bench.md)、[ESI-Bench（具身空间智能）](../entities/esi-bench.md)、[EWMBench（世界模型生成评测）](../entities/ewmbench.md)、[GigaWorld-1（策略评估器）](../entities/paper-gigaworld-1-policy-evaluation.md)、[仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)。它是[具身大模型分类学选型闭环](../queries/embodied-fm-taxonomy-loop.md)（选哪一类模型）的姊妹链——回答「选完之后怎么评测/证明它」。

## TL;DR：四层评测选型闭环一句话定位

「评测一个具身大模型」不是「跑一个成功率就完事」，而是一条**从认知到执行、再由 sim↔real 校准兜底的分层评测链**。每一层测的对象、代表性基准、指标语义都不同，**上一层评分高不代表下一层能成**——认知评分 ≠ 可执行动作，视频逼真 ≠ 策略收益，仿真成功率 ≠ 真机成功率。选错评测层，就会用「测得漂亮的假信号」掩盖「真正会崩的地方」：

| 层 | 测什么 | 代表性基准 | 主指标 | 这一层评测最容易骗人的地方 |
|----|--------|-----------|--------|--------------------------|
| ① 具身大脑/MLLM 认知 | System 2 高层认知：意图理解、场景感知、规划、affordance、失败诊断 | [RoboBench](../entities/robo-bench.md)、[ESI-Bench](../entities/esi-bench.md) | QA 正确率 / 认知维度分 | 认知评分高 ≠ 能下发可执行动作 |
| ② 世界模型预测保真度 | 给定动作，模型能否忠实推演未来帧/物理状态 | [EWMBench](../entities/ewmbench.md)、[GigaWorld-1 / WMBench](../entities/paper-gigaworld-1-policy-evaluation.md) | 场景守恒 / 轨迹一致 / 语义对齐 | 短时视觉逼真 ≠ 长时序动作忠实 ≠ 下游策略收益 |
| ③ 策略任务成功率 | 策略在任务上真做成没有 | [ManiSkill-HAB](../entities/paper-notebook-maniskill-hab-a-benchmark-for-low-level-manipula.md)、[Mimicking-Bench](../entities/paper-notebook-mimicking-bench-a-benchmark-for-generalizable-hu.md)、[Barkour](../entities/paper-barkour-quadruped-agility-benchmark.md)；桌面 VLA 相对位次见 [VLA SOTA Leaderboard](../entities/vla-sota-leaderboard.md) | 任务成功率 / 敏捷分 | 成功率均值掩盖长尾失败；魔法抓取虚高；跨基准直接比榜 |
| ④ sim↔real 评测 gap 校准 | 仿真评测结论能否外推到真机 | [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md) + real-to-sim 相关性 | sim↔real 排名相关性 | 仿真可复现 ≠ 真机代表性；评测集与训练分布重叠 |

**总原则**：评测选型的第一问永远是「**这层指标测的到底是能力本身，还是能力的易测代理**」。越靠上层（认知、视频质量）越好测、越可复现，但离「真机做成」越远；越靠下层（真机成功率、sim↔real 校准）越贵、越难复现，但代表性越强。一条负责任的评测链要**逐层往下压实**，而不是停在某个漂亮的上层代理指标上。

---

## 四层评测选型决策树

```mermaid
flowchart TD
  start[要评测一个具身大模型: 从哪层测起?]
  start --> l1{① 先证 System 2 大脑认知合格吗?}
  l1 -->|是 · MLLM 高层认知| brain[RoboBench: 意图/感知/规划/affordance/失败诊断<br/>ESI-Bench: 主动探索式空间智能]
  l1 -->|大脑已达标| l2
  brain --> l2{② 用世界模型当评估器/前瞻吗?}
  l2 -->|是 · 视频 WM 保真度| wm[EWMBench: 场景守恒/轨迹/语义对齐<br/>GigaWorld-1: 长时序动作忠实 rollout]
  l2 -->|否 · 直接上真机/仿真跑策略| l3
  wm --> l3{③ 策略成功率在哪测?}
  l3 -->|仿真高吞吐可复现| simeval[ManiSkill-HAB 低层操作<br/>Mimicking-Bench 人形模仿<br/>Barkour 四足敏捷]
  l3 -->|真机代表性优先| realeval[真机 rollout: 贵/慢/难复现<br/>但接触/感知噪声/长尾最真实]
  simeval --> l4{④ 仿真结论能外推真机吗?}
  realeval --> l4
  l4 -->|需校准| gap[real-to-sim 相关性校准<br/>见 sim↔real 评测 gap 概念页]
  l4 -->|已校准/可外推| ship[评测结论可用于选型/发版决策]
  gap --> ship
```

---

## 1. ① 具身大脑/MLLM 认知评测层：先证「大脑」合格

整条评测链的入口是**在双系统范式下把 MLLM 当 embodied brain 单独考核**，把「机器人最终能不能做成」拆出「System 2 是否具备操纵所需的完整高层认知」这一前置问题：

- **测什么**：意图理解 → 场景感知 → 规划与泛化 → affordance 细化 → 失败诊断的全流水线认知。[RoboBench](../entities/robo-bench.md) 沿五维 14 能力 25 任务 6092 QA 出题，并用 **MLLM-as-world-simulator** 检验规划是否能在物理/视觉约束下达成关键物体状态变化；[ESI-Bench](../entities/esi-bench.md) 进一步把空间智能从「被动看图」推进到「观察者即行动者」，要求主动感知–行动闭环。
- **可复现 vs 代表性**：QA 式评分**高度可复现、可自动打分**，是四层里最便宜的一层；代价是它测的是**认知代理**而非动作能力。
- **典型误判**：把认知评分当动作能力用——RoboBench 的价值恰恰在于它证明了「认知分与 CALVIN/LIBERO-10 下游 VLA 成功率显著相关」这件事**需要专门验证**，而不是默认成立。认知评分只是**下游成功率的必要不充分条件**。

## 2. ② 世界模型预测保真度评测层：视频逼真 ≠ 策略收益

当团队用世界模型做前瞻推演或当策略评估器时，必须先评测 **WM 本身预测得准不准**，否则「用一个不忠实的 WM 去评策略」会双重放大误差：

- **测什么**：给定初始帧 + 指令（及可选动作序列），模型自回归续写未来帧，评测其**场景守恒、末端轨迹正确性、语义/逻辑对齐**（[EWMBench](../entities/ewmbench.md) 在 Agibot-World 子集上统一初始化后三轴打分）。
- **过程 vs 结果的关键结论**：[GigaWorld-1](../entities/paper-gigaworld-1-policy-evaluation.md) 在 WMBench 上用 7 类视频 WM × 4 种动作编码 × 32.4 万+ rollout 实证——**长时序动作忠实一致性比短时视觉逼真更决定 evaluator 质量**。这直接推翻了「视频看起来越真、当评估器越好」的直觉。
- **典型误判**：① 用短时视觉逼真度（FVD 类）代表长时序动作忠实度；② 把 WM 的视频质量当成下游策略收益——视频质量是**过程/中间指标**，策略成功率才是**结果指标**，二者不可互相替代。

## 3. ③ 策略任务成功率评测层：均值成功率的陷阱

到这一层才第一次直接测「策略做成没有」，但**成功率这个结果指标本身也有可信度分层**：

- **测什么/用什么基准**：[ManiSkill-HAB](../entities/paper-notebook-maniskill-hab-a-benchmark-for-low-level-manipula.md) 用**真实低层控制**替代「魔法抓取」测家庭重排（GPU 加速、可控演示生成）；[Mimicking-Bench](../entities/paper-notebook-mimicking-bench-a-benchmark-for-generalizable-hu.md) 用大规模人类技能参考系统比较重定向/跟踪/模仿学习组合，测人形全身交互技能泛化；[Barkour](../entities/paper-barkour-quadruped-agility-benchmark.md) 用犬敏捷赛式障碍课 + 0–1 敏捷分测四足敏捷性。桌面语言条件 VLA 的社区相对位次可先扫 [VLA SOTA Leaderboard](../entities/vla-sota-leaderboard.md)（LIBERO / Meta-World / RoboTwin 等**摘录分数**，不重跑），再回原文核协议——例如 [FabriVLA](../entities/paper-fabrivla.md) 的 MT50 **90.0%** 与 [Evo-1](../entities/paper-evo1-lightweight-vla.md) 的 **80.6%** 同属该层，但训练配方与评测面不同。
- **可复现 vs 代表性**：仿真成功率**高吞吐、可复现、可控**，适合 recipe 迭代；但「魔法抓取」这类抽象化实现会**系统性虚高**成功率——ManiSkill-HAB 的意义正是把重排基准**落到真实低层操作**上，缩小这道代表性缺口。榜站聚合视图**不能替代**官方评测脚本与协议脚注。
- **典型误判**：① **成功率均值掩盖长尾失败模式**——同样 80% 成功率，均匀失败 vs 集中在某类物体/初值上的失败，工程含义天差地别；② 单任务过拟合冒充跨任务泛化，需 Mimicking-Bench 这类**跨任务/跨物体**基准把关；③ 离线回放评测（固定初值重放）≠ 在线闭环评测（策略自己滚出轨迹），后者才暴露复合误差；④ **跨基准直接比榜**（LIBERO vs Meta-World vs RoboTwin）——[VLA SOTA Leaderboard](../entities/vla-sota-leaderboard.md) Methodology 明确禁止。

## 4. ④ sim↔real 评测 gap 校准层：评测结论能否外推真机

前三层多数在仿真里完成，最后必须回答**「仿真里测出来的结论，能不能外推到真机」**——否则再漂亮的仿真榜单也只是自证：

- **测什么**：不是再测一次策略，而是测**评测本身的可外推性**——[仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)把可信仿真当作**可扩展闭环评测引擎**，其前提是仿真 rollout 与真机 rollout **统计相关**，且训练管线**刻意不与评测共享同一仿真分布**（避免评测集泄漏）。
- **为什么必须单独一层**：仿真在可复现性/吞吐/可控性上的优势，是**以牺牲真实接触、感知噪声、长尾分布代表性**换来的。这条 gap 的物理根因与三条缩小路线，单独沉淀为姊妹概念页 [仿真评测可复现性 ↔ 真实代表性取舍（sim↔real 评测 gap）](../concepts/sim-vs-real-eval-gap.md)。
- **典型误判**：① 仿真基准饱和（刷到接近满分）当成「真实场景就绪」；② 评测集与训练分布重叠导致虚高（数据泄漏）；③ 静态基准不覆盖部署时的分布漂移。校准手段是**用少量真机 rollout 锚定 sim↔real 排名相关性**，而非用仿真绝对分。

---

## 评测层选型矛盾速查（按取舍归因）

| 矛盾 | 一端 | 另一端 | 选型第一判据 |
|------|------|--------|-------------|
| 可复现 vs 代表性 | 仿真基准高吞吐可复现 | 真机 rollout 代表性强 | 结论要外推真机就必须校准 gap |
| 过程 vs 结果指标 | 视频质量/认知分好测 | 任务成功率才是收益 | 上层分只是下游成功率的代理 |
| 均值 vs 长尾 | 平均成功率一个数 | 长尾失败模式分布 | 是否关心最坏情形/安全 |
| 离线 vs 在线 | 回放固定初值可复现 | 闭环滚动暴露复合误差 | 部署是闭环就必须在线评 |
| 单任务 vs 跨任务 | 单任务榜单好刷 | 跨任务/跨物体泛化 | 是否宣称通用能力 |

## 典型失败模式速查（按评测层归因）

| 现象 | 最可能测错的评测层 | 第一优先排查 |
|------|------------------|-------------|
| 认知榜单高但真机不动 | ① 拿认知分当动作能力 | 补 ③ 真机成功率闭环评测 |
| WM 视频很真但选出的策略更差 | ② 用视觉逼真代替动作忠实 | 换长时序动作忠实指标（GigaWorld-1） |
| 仿真成功率高真机崩 | ③/④ 魔法抓取虚高 / sim↔real 未校准 | 落到真实低层控制 + real-to-sim 相关性 |
| 平均成功率好但偶发大事故 | ③ 均值掩盖长尾 | 按失败模式/物体分层看成功率 |
| 榜单饱和但新场景失效 | ④ 基准饱和 ≠ 场景就绪 / 分布漂移 | 换分布外测试集，查评测集泄漏 |

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MLLM | Multimodal Large Language Model | 多模态大语言模型，②层充当具身大脑 System 2 |
| WM | World Model | 世界模型，②层预测未来状态/帧的评测对象 |
| WMBench | World Model Benchmark | GigaWorld-1 提出的世界模型策略评估器基准 |
| WMES | World Model Evaluation Score | GigaWorld-1 的世界模型评估综合分 |
| EWM | Embodied World Model | 具身世界模型，EWMBench 的评测对象 |
| VLA | Vision-Language-Action | 视觉–语言–动作端到端策略，③层被评对象 |
| HAB | Home Assistant Benchmark | 家庭助理基准，ManiSkill-HAB 的评测场景 |
| sim2real | Simulation-to-Real | 仿真到真机迁移，④层校准 gap 的核心 |

## 参考来源

- [robo_bench_arxiv_2510_17801.md](../../sources/papers/robo_bench_arxiv_2510_17801.md) — RoboBench，①层 MLLM 具身大脑五维认知评测与「认知分↔下游成功率相关」实证
- [ewmbench.md](../../sources/papers/ewmbench.md) — EWMBench，②层具身世界模型视频生成三轴评测
- [esi_bench_arxiv_2605_18746.md](../../sources/papers/esi_bench_arxiv_2605_18746.md) — ESI-Bench，①层主动探索式具身空间智能评测
- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md) — GigaWorld-1「长时序动作忠实 > 短时视觉逼真」策略评估器结论

## 关联页面

- 所属专题：[具身评测基准选型闭环（专题汇总）](../overview/topic-embodied-eval-benchmark.md) — 四层评测基准的统一入口与图谱专题枢纽
- [仿真评测可复现性 ↔ 真实代表性取舍（sim↔real 评测 gap）](../concepts/sim-vs-real-eval-gap.md) — ④层 gap 校准的姊妹概念页，双向回链
- [RoboBench（MLLM 具身大脑综合评测）](../entities/robo-bench.md) — ①层 MLLM 认知评测代表基准
- [ESI-Bench（具身空间智能基准）](../entities/esi-bench.md) — ①层主动探索式空间智能评测
- [EWMBench（具身世界模型生成评测）](../entities/ewmbench.md) — ②层世界模型预测保真度评测
- [GigaWorld-1（世界模型策略评估器）](../entities/paper-gigaworld-1-policy-evaluation.md) — ②层「动作忠实 > 视觉逼真」策略评估器
- [DriftWorld](../entities/paper-driftworld.md) — ②层外延：1-step drifting 快评估 + 推理时搜索（相关性最高约 0.99）
- [Masked Visual Actions](../entities/paper-masked-visual-actions.md) — ②层外延：掩码动作条件 WM，RoboCasa 策略评估 **r=0.982**
- [Ctrl-World](../entities/paper-ctrl-world.md) — ②层外延：多视角可控 WM，VLA 想象评估 + 合成轨迹改进（ICLR 2026）
- [VLA SOTA Leaderboard](../entities/vla-sota-leaderboard.md) — ③层社区聚合：多基准 VLA / 灵巧手摘录榜（不重跑）
- [FabriVLA](../entities/paper-fabrivla.md) — ③层轻量 VLA Meta-World 对照条目
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md) — ④层可信仿真作闭环评测引擎的前提
- [Sim2Real](../concepts/sim2real.md) — ④层评测结论外推真机的迁移背景
- 姊妹 Query：[具身大模型分类学选型闭环](../queries/embodied-fm-taxonomy-loop.md) — 「选哪一类模型」，本页承接「选完怎么评测」
