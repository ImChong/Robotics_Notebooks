# fact-wam.github.io（FACT 项目页）

- **标题：** FACT: Failure-Aware Causal Training for World-Action Models
- **类型：** site / project-page
- **URL：** <https://fact-wam.github.io/>
- **配套论文：** [FACT（arXiv:2608.10232）](https://arxiv.org/abs/2608.10232) — 归档见 [`sources/papers/fact_arxiv_2608_10232.md`](../papers/fact_arxiv_2608_10232.md)
- **代码：** <https://github.com/Bariona/FACT> — [`sources/repos/fact.md`](../repos/fact.md)
- **权重：** <https://huggingface.co/Bariona/fact-wam>
- **入库日期：** 2026-08-13

## 一句话摘要

UCSD 官方站点：展示因果 WAM「先动后想」、失败轨迹作后果监督、可选 value 打分；给出 RoboTwin 与真机双臂结果，以及失败想象对照视频。

## 公开信息要点（截至入库日）

- **入口：** Paper / Code / Model 链接齐全（步骤 2.5 → **已开源**）。
- **方法叙事：** Act then imagine；failures teach consequences；optional best-of-N scoring。
- **仿真：** RoboTwin 50 任务平均含失败共训 **87.5%**；部署约 **3×** 快于最强对照档。
- **真机：** seen **89%**（+scoring **92%**）；失败数据 scaling 未早饱和；坏动作未来 PSNR **+6.4 dB**。

## 为何值得保留

- **步骤 2.5 开源核查主入口：** Code + HF Model 可点。
- **失败感知 WAM 选型读点：** 与「只堆成功演示」路线对照。

## 关联资料

- 论文归档：[`sources/papers/fact_arxiv_2608_10232.md`](../papers/fact_arxiv_2608_10232.md)
- 代码归档：[`sources/repos/fact.md`](../repos/fact.md)
- Wiki 实体：[wiki/entities/paper-fact.md](../../wiki/entities/paper-fact.md)
