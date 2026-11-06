# When AI builds itself 通读（Anthropic Institute）

> 来源归档（原文通读 · 支撑 `wiki/concepts/recursive-self-improvement.md`）

- **标题：** When AI builds itself — Our progress toward recursive self-improvement, and its implications
- **作者：** Marina Favaro, Jack Clark（Santi Ruiz 编辑）
- **类型：** site / institute essay
- **原文：** <https://www.anthropic.com/institute/recursive-self-improvement>（2026-11 通过 HTML 抓取全文，含脚注与 2026-09-18 更新注记）
- **一句话说明：** 用公开基准与内部未发表数据论证 AI 已在加速 AI 研发；完全 RSI 尚未到来、也非必然，并给出三个情景与「可信放慢需可验证」的政策立场。
- **核对范围：** 通读正文全部章节（公开证据、内部证据、人类角色变化、"What if we're wrong"、三个情景、"What should we do"）及全部 9 条脚注与致谢；文中图表为图片，仅依据其正文说明与 "How to read this" 注释，未读取图中原始数据点；内部数字为公司自报，无法外部复现。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定义（开篇）：** 足够算力下，AI 能自主设计并开发后继者即递归自改进；「尚未到达，且非必然」。
- **公开证据：** METR 50% 可靠任务时长约每 4 个月翻倍（此前约 7 个月）；Opus 3 约 4 分钟 → Sonnet 3.7 约 1.5 小时 → Opus 4.6 约 12 小时；Mythos Preview「至少」16 小时。SWE-bench 两年内饱和；CORE-Bench 由约 20%（2024）15 个月后饱和。脚注 2：基准常因题目错误饱和于 100% 以下。
- **内部证据：** 2026-05 合入代码 >80% 由 Claude 撰写（脚注 3：领导层口述 90%+，>80% 更保守）；2026 Q2 工程师人均日合入量约 2024 的 8×，作者承认 LOC 高估真实生产率；2026-03 对 130 名研究员问卷中位自评约 4×，作者认为真实值偏低（脚注 6 引 METR 研究）。
- **开放任务：** 最开放难度档成功率 2026-05 为 76%，半年 +50 个百分点（Claude judge 评判）。
- **实验微型环：** 固定正确性约束下加速小模型训练代码，Opus 4（2025-05）约 3× → Mythos Preview（2026-04）约 52×；熟练人类 4–8 小时约 4×；脚注 7 强调绝对值依赖起点代码，不是真实训练加速。
- **端到端研究：** 弱监督强模型任务，两名人类约一周收回 23% 差距，agents 800 累计小时、约 1.8 万美元算力收回 97%；未干净迁移到生产规模，题目与评分仍由人定。
- **研究判断：** 129 个人类走弯路的会话时刻，Opus 4.5 优于人类选择 51%，Mythos Preview 64%；对照 127 个人类选择已较强的时刻，模型仅约 20% 更优（脚注 8）。
- **人的角色：** 「做」几乎零人力成本，研究品味与判断（选问题、信结果、判死路）是人的比较优势；人工代码审查成为新瓶颈（Amdahl 定律）。
- **反驳与推论（What if we're wrong）：** 进展多为增量工作而非「顿悟」；保守读法也意味着复合加速；宽松读法认为研究品味可能只是又一项会被攻克的能力。
- **三个情景：** (1) 趋势失速但能力扩散（作者认为不太可能；Glasswing 找到万余高危漏洞，瓶颈转向修补）；(2) 复合效率、人仍定方向（作者认为很可能正在进入）；(3) 完全 RSI（进度由算力与算法效率决定，对齐走向最不确定，具身智能预期会跟随，但药监、选举、信任等瓶颈仍按人类时间）。
- **政策：** 愿意在可验证的多边放慢条件下暂停；训练比导弹井更难探测；单边暂停只改变领跑者；Institute 将组织对话。

## 对 wiki 的映射

- [`wiki/concepts/recursive-self-improvement.md`](../../wiki/concepts/recursive-self-improvement.md)
