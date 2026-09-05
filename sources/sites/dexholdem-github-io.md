# dexholdem.github.io/Dexholdem（DexHoldem 项目页）

- **标题：** DexHoldem: Playing Texas Hold'em with Dexterous Embodied System
- **类型：** site / project-page
- **URL：** <https://dexholdem.github.io/Dexholdem/>
- **配套论文：** [arXiv:2605.18727](https://arxiv.org/abs/2605.18727) — [`sources/papers/dexholdem_arxiv_2605_18727.md`](../papers/dexholdem_arxiv_2605_18727.md)
- **代码：** [DexHoldem/Dexholdem-Policy](https://github.com/DexHoldem/Dexholdem-Policy) — [`sources/repos/dexholdem-policy.md`](../repos/dexholdem-policy.md)；技能仓 [DexHoldem/DexHoldemSKills](https://github.com/DexHoldem/DexHoldemSKills) — [`sources/repos/dexholdem-skills.md`](../repos/dexholdem-skills.md)
- **数据集：** [Winniechen2002/TexasPokerRobot](https://huggingface.co/datasets/Winniechen2002/TexasPokerRobot) — [`sources/datasets/texaspokerrobot.md`](../datasets/texaspokerrobot.md)
- **入库日期：** 2026-09-05

## 一句话摘要

港大 IDS 团队的真机 ShadowHand + UR10e 德州扑克桌面基准：1,470 条遥操作示范、14 个扑克原语、36 题具身感知题，以及感知→路由→灵巧策略的闭环案例。

## 公开信息要点（截至 2026-09-05 复核）

- **页首：** *DexHoldem: Playing Texas Hold'em with Dexterous Embodied System*；作者 Feng Chen\*†、Tianzhe Chu\*、Li Sun\*、Pei Zhou\*、Zhuxiu Xu、Shenghua Gao、Yuexiang Zhai、Yanchao Yang、Yi Ma。
- **步骤 2.5：** Resources 区同时给出 Paper PDF、Hugging Face 数据集、GitHub organization、Policy 仓、Skills 仓、演示视频与投稿入口。→ **已开源**（策略训练/部署代码 + 数据集 + agent skill；两仓截至入库日未附 LICENSE 文件）。
- **三块评测：** Policy Bench（9 个策略 × 14 原语 × 各 80 次真机试）；Perception Bench（36 题结构化状态恢复）；System-level（3 条 GPT 5.5 + π₀ 整手案例）。
- **当前榜：** 策略最优 TCR **61.2%** / SPSR **47.5%**（π₀.₅）；感知最优 field-wise **66.8%**（GPT 5.5）、严格 exact match **34.3%**（Opus 4.7）。评测日期页上写 2026-05-07。
- **系统叙事：** 把编码 agent harness（Claude Code / Codex / Gemini CLI）当成 perceiver+router，确定性 `router.py` 管等待/恢复/求助，再把 agent primitive 翻译成 30 维关节策略原语。

## 为何值得保留

- 步骤 2.5 的项目页入口；真机数字与视频比 PDF 更完整。
- 把「灵巧原语做成」和「牌桌状态读对」拆成两张榜，避免把扑克智能误读成操作评测。

## 关联资料

- 论文：[`sources/papers/dexholdem_arxiv_2605_18727.md`](../papers/dexholdem_arxiv_2605_18727.md)
- 策略仓：[`sources/repos/dexholdem-policy.md`](../repos/dexholdem-policy.md)
- 技能仓：[`sources/repos/dexholdem-skills.md`](../repos/dexholdem-skills.md)
- 数据集：[`sources/datasets/texaspokerrobot.md`](../datasets/texaspokerrobot.md)
- Wiki：[wiki/entities/paper-dexholdem.md](../../wiki/entities/paper-dexholdem.md)
