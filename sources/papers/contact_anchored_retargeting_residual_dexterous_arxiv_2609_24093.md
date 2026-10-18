# Contact-Anchored Retargeting（arXiv:2609.24093）

> 来源归档（paper）

- **标题：** Dexterous Robot Manipulation from Human Demonstrations via Contact-Anchored Retargeting and Residual Policy Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.24093>
- **PDF：** <https://arxiv.org/pdf/2609.24093>
- **代码：** <https://github.com/DexGEM-Lab/real2sim2real>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 先用物理残差学习补全人类示范接触，再按接触位置和顺序重定向到机器人手，最后修复动力学可行性。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 跨手形态迁移要先保留接触结构，而不是匹配关节。
- 残差策略补全接触 → 按接触位置与顺序重定向 → 动力学可行性修复。
- 指标：公众号称覆盖四种手形态与真机双手任务；训练仍依赖仿真。

## 对 wiki 的映射

- [paper-contact-anchored-retargeting-residual-dexterous](../../wiki/entities/paper-contact-anchored-retargeting-residual-dexterous.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
