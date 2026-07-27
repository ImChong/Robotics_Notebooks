# RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies

> 来源归档（ingest）

- **标题：** RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies
- **类型：** paper / benchmark / manipulation / sim2real / evaluation / vla
- **arXiv abs：** <https://arxiv.org/abs/2607.04434>
- **提交日期：** 2026-07-05（v3 updated 2026-07-08）
- **项目页：** <https://robodojo-benchmark.com/>
- **代码：** <https://github.com/RoboDojo-Benchmark/RoboDojo>
- **策略基建：** <https://github.com/XPolicyLab/XPolicyLab>
- **榜单：** <https://robodojo-benchmark.com/leaderboard>
- **机构 / 运营：** AI MMLab Club（非营利）+ 全球学术机构联盟（论文作者跨多机构；治理见官网 README Affiliations）
- **作者：** Tianxing Chen、Yue Chen、Zixuan Li、Junyuan Tang、Kailun Su、Haoran Lu、Weijie Wan、Baijun Chen 等（共约 44 人）
- **入库日期：** 2026-07-27
- **一句话说明：** 提出 **RoboDojo** 统一仿真–真机操纵评测：**42** 仿真任务（五维：泛化 / 记忆 / 精度 / 长程 / 开放词汇）+ **18** 真机任务；Isaac Sim **异构并行**；**RoboDojo-RealEval** 远程标准化真机评测；与 **XPolicyLab** 一次集成跨 sim/real；论文称集成 **30** 策略并建公共榜单（仓内适配已扩至 40+，见 repos）。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 论文 | <https://arxiv.org/abs/2607.04434> | arXiv 2607.04434 |
| 官网 | <https://robodojo-benchmark.com/> | 文档 / 榜单 / Eval |
| 代码 | <https://github.com/RoboDojo-Benchmark/RoboDojo> | eval-only 仿真评测栈 |
| XPolicyLab | <https://github.com/XPolicyLab/XPolicyLab> | 策略训推与上榜产物发布口 |
| 关联工作 | RoboTwin 2.0 (arXiv:2506.18088)、MagicSim (arXiv:2606.17511) | README citation |

## 核心摘录（面向 wiki 编译）

### 1) 问题动机：窄技能基准 + 仅 sim 或仅 real

- **摘录要点：** 现有通用操纵评测常依赖短程、技能狭窄、模式相似任务，能力维度覆盖不足；且多只在仿真或只在真机进行——仿真可扩展但缺物理部署挑战，真机代表性强但贵、慢、难复现。
- **对 wiki 的映射：**
  - [RoboDojo](../../wiki/entities/robodojo.md) — 统一 sim-and-real 的定位。
  - [仿真 vs 真机评测 gap](../../wiki/concepts/sim-vs-real-eval-gap.md) — 第四层校准语境。

### 2) 五维仿真能力 + 真机部署压力

- **摘录要点：** 仿真评测五维 — **generalization、memory、precision、long-horizon execution、open-vocabulary instruction following**；真机基准将策略暴露于具挑战性的物理部署条件；规模 **42 sim + 18 real**。
- **对 wiki 的映射：**
  - [RoboDojo](../../wiki/entities/robodojo.md) — 能力维与任务规模表。

### 3) 异构并行仿真 + RoboDojo-RealEval

- **摘录要点：** Isaac Sim 上异构并行（不同任务/场景/进程并发）提供可扩展反馈；RealEval 提供远程云访问、标准化硬件、场景复位、评测协议与部署接口，以提升真机评测可复现性。
- **对 wiki 的映射：**
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
  - [RoboDojo](../../wiki/entities/robodojo.md) — 工程实践与流程总览。

### 4) XPolicyLab：一次集成，sim/real 同接口

- **摘录要点：** 与 XPolicyLab 配合，策略一次适配即可在仿真与真机评测间最小改动切换；论文报告集成 **30** 策略并建立公共 leaderboard 与系统分析。
- **对 wiki 的映射：**
  - [XPolicyLab](../../wiki/entities/xpolicylab.md)
  - [VLA](../../wiki/methods/vla.md) — 通用策略评测入口。

### 5) 与长期公益上榜规则的关系（官网补充，非论文正文）

- **摘录要点：** 2026-07 官网开放长期公益评测入口：verified 上榜须官方云评测 + 隐藏布局校验；分数公布前在 XPolicyLab 公开训推代码与权重，并发布评测视频（见 [站点归档](../sites/robodojo-benchmark.md)、[公告](../blogs/robodojo_open_longterm_eval_2026-07.md)）。
- **对 wiki 的映射：**
  - [RoboDojo](../../wiki/entities/robodojo.md) — 「上榜与开源协议」节。

## 当前提炼状态

- [x] arXiv 摘要、任务规模、五维、RealEval、XPolicyLab 分工已摘录
- [x] 与 `sources/repos/robodojo.md`、`xpolicylab.md`、`sites/robodojo-benchmark.md` 分工明确
- [x] wiki 映射：`wiki/entities/robodojo.md`、`wiki/entities/xpolicylab.md` 新建并交叉具身评测选型链
