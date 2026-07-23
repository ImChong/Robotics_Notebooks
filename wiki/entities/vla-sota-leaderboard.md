---
type: entity
tags: [leaderboard, benchmark, vla, dexterous-manipulation, open-source, sjtu, evomind, meta-world, libero, calvin, robotwin]
status: complete
updated: 2026-07-23
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../overview/vla-open-source-repro-landscape-2025.md
  - ../overview/topic-vla.md
  - ./paper-evo1-lightweight-vla.md
  - ./paper-fabrivla.md
  - ./paper-mint-vla.md
sources:
  - ../../sources/sites/sota-evomind-tech.md
  - ../../sources/repos/evo-sota-io.md
summary: "EvoMind & MINT-SJTU 维护的 VLA / 灵巧手多基准 SOTA 排行榜（sota.evomind-tech.com）：从论文摘录分数、按开源与协议分类，覆盖 LIBERO / Meta-World / CALVIN / RoboTwin 2.0 等与 Adroit 等灵巧手榜；MIT 开源静态站。"
---

# VLA SOTA Leaderboard（EvoMind / MINT-SJTU）

**VLA SOTA Leaderboard**（线上站 [sota.evomind-tech.com](https://sota.evomind-tech.com/)，源码 [MINT-SJTU/Evo-SOTA.io](https://github.com/MINT-SJTU/Evo-SOTA.io)，MIT）由 **进化思维科技（EvoMind Tech）** 与 **上海交通大学 MINT-SJTU** 社区维护：把已发表论文与官方仓库中的 **Vision-Language-Action** 与 **灵巧手操作** 分数汇总成可排序、可过滤的榜单，并区分开源/闭源与 SFT/RL 训练范式——**不重跑实验**，定位是**研究导航与相对位次索引**，而非官方认证排名。

## 一句话定义

**从公开论文摘录多基准 VLA（及灵巧手）分数，按开源状态与评测协议分类展示的社区排行榜与静态站。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| SOTA | State of the Art | 当前公开报道的最佳水平（榜站语义） |
| LIBERO | Lifelong Robot Learning Benchmark | 语言条件终身/多任务操作基准族 |
| MT50 | Meta-World 50 Tasks | Meta-World 五十任务多任务操作套件 |
| CALVIN | Composing Actions from Language and Vision | 长程语言条件桌面操作基准 |
| SFT | Supervised Fine-Tuning | 监督微调；榜站可与 RL 条目对照过滤 |

## 为什么重要

- **选型加速：** 在 [VLA 方法页](../methods/vla.md) 与各论文实体之间，提供「先看相对位次 → 再回原文核协议」的入口。
- **开源可发现性：** 默认突出带 Open Source 徽章的模型；闭源 / Coming Soon 需显式展开，降低误以为可复现的风险。
- **协议意识：** Methodology 明确 **跨基准不可比**、协议差异会导致数字漂移——与 [具身评测选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) 第 ③ 层「均值成功率陷阱」互补。

## 核心原理

### 覆盖范围

| 轨道 | 基准（主指标） |
|------|----------------|
| **VLA** | LIBERO、LIBERO Plus、Meta-World、CALVIN（主看 ABC→D Avg. Len.）、RoboTwin 2.0（Hard SR）、RoboChallenge（Score）、RoboCasa-GR1-Tabletop |
| **Dex**（`/dex/`） | Adroit、DexArt、Bi-DexHands |

### 数据与排名逻辑

```mermaid
flowchart LR
  papers["已发表论文 / 官方仓"] --> extract["人工摘录分数"]
  extract --> json["public/data/*.json\nopensource / closed / non_standard"]
  json --> ui["Next.js 静态站\n排序 · 过滤 · 搜索"]
  ui --> reader["读者回原文核协议"]
```

- **不重跑：** 分数来自原文或第三方复现报道；站点免责声明要求引用前核实。
- **分区：** `standard_opensource` / `standard_closed` / `non_standard`（非标准协议等）。
- **过滤：** 开源徽章、RL vs SFT、中英界面；`/models/` 模糊搜索跨榜定位。

### 入库日 Meta-World 快照（开源标准区）

> 数字以站点 `data/metaworld.json` 为准；会随维护更新。

| 模型 | average | 备注 |
|------|---------|------|
| [FabriVLA](./paper-fabrivla.md) | **90.0** | 开源区 rank 1（入库日） |
| LA4VLA-1B | 87.53 | 同榜前列 |
| [Evo-1](./paper-evo1-lightweight-vla.md) | **80.6** | 同维护生态代表作 |

## 工程实践

| 项 | 要点 |
|----|------|
| **本地预览** | `git clone` → `npm install` → `npm run dev`（Node 18+） |
| **生产构建** | `npm run build` → `out/`（GitHub Pages） |
| **纠错 / 投稿** | GitHub Issues 或 `business@evomind-tech.com`；提交论文链接与分数 |
| **站内用法** | 选榜 → 看开源过滤 → 点 paper/code URL → 回本库实体页对照训练配方 |

## 局限与风险

- **误区：** 把榜分数当成「官方 SOTA 认证」或跨 LIBERO / Meta-World / RoboTwin **直接比大小**。
- **误区：** 忽略 `non_standard` 与评测协议脚注——同名基准下设定差可导致虚高/虚低。
- **局限：** 站点不替代各基准官方评测脚本；CALVIN 等分区可能阶段性为空或滞后。
- **工程风险：** 静态 JSON 滞后于最新 arXiv；应以原文与官方仓为准更新站内 wiki。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md) — 方法主线与轻量/通才路线索引。
- [FabriVLA](./paper-fabrivla.md) — Meta-World 轻量 VLA 开源代表（入库日榜首区）。
- [Evo-1](./paper-evo1-lightweight-vla.md) — 同生态轻量 VLA；LeRobot 集成。
- [具身评测选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 分层评测决策；本榜属第 ③ 层聚合视图。
- [VLA 开源复现景观](../overview/vla-open-source-repro-landscape-2025.md) — 按复现目标选仓库。
- [Manipulation](../tasks/manipulation.md) — 桌面操作任务语境。

## 参考来源

- [VLA SOTA Leaderboard 站点归档](../../sources/sites/sota-evomind-tech.md)
- [MINT-SJTU/Evo-SOTA.io 仓库归档](../../sources/repos/evo-sota-io.md)

## 推荐继续阅读

- 线上站：<https://sota.evomind-tech.com/>
- Methodology：<https://sota.evomind-tech.com/methodology/>
- 源码与贡献说明：<https://github.com/MINT-SJTU/Evo-SOTA.io>
