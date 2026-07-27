# RoboDojo 开放长期公益评测（社区公告）

> 来源归档

- **标题：** #RoboDojo开放长期公益评测
- **类型：** blog / announcement（社区开放评测入口通知）
- **关联官网：** <https://robodojo-benchmark.com/>
- **榜单：** <https://robodojo-benchmark.com/leaderboard>
- **上榜入口与规则：** <https://robodojo-benchmark.com/eval>
- **代码：** <https://github.com/RoboDojo-Benchmark/RoboDojo>
- **XPolicyLab：** <https://github.com/XPolicyLab/XPolicyLab>（集成 40+ 前沿模型复现）
- **日期：** 2026-07（公告日；站内 protocol 时间戳约 2026-07-27）
- **入库日期：** 2026-07-27
- **一句话说明：** 正式开放 RoboDojo **线上评测上榜入口与规则**；榜单由国内外学术机构 **公益运行**；为保障公正，分数对外公布前须在 **XPolicyLab** 公开模型训推代码（支持复现与社区监督）并 **开源模型权重**，同时公布 **评测视频**。

## 公告原文要点（策展）

1. 开放 **线上评测上榜入口与规则**（Eval）与 **Leaderboard**。
2. 榜单由 **全学术机构公益运行**（与官网「AI MMLab Club + 学术共治、无商业资助」一致）。
3. **公正性门槛：** 分数公开前 — 在 XPolicyLab 公开训推代码 + 开源权重；公布评测视频；欢迎社区共建评测生态。
4. 外链齐全：官网 / Eval / 代码仓 / XPolicyLab。

## 与官网 Protocol 对齐（核查）

公告口径与 [leaderboard/protocol](https://robodojo-benchmark.com/leaderboard/protocol) 一致，且官网更细：

- Private 远程评测可不先开源；**verified** 公布阶段才强制完整评测产物（训推代码、checkpoint、配置、推理与部署说明）经 XPolicyLab 释放。
- 另有：官方云评测管线、三 seed 仿真统计、三真机本体覆盖、hidden-layout 校验。

## 对 wiki 的映射

- [RoboDojo](../../wiki/entities/robodojo.md) — 「长期公益评测与上榜规则」主节
- [XPolicyLab](../../wiki/entities/xpolicylab.md) — 上榜产物发布与 40+ 复现
- [具身评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — 策略成功率层 + sim↔real 层代表基准
