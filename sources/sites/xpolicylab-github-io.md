# XPolicyLab 项目页

> 来源归档

- **标题：** XPolicyLab — Unified standard and open ecosystem for robot policy evaluation and deployment
- **类型：** site / project-page
- **URL：** <https://xpolicylab.github.io/>
- **论文：** <https://arxiv.org/abs/2608.09892> — [`sources/papers/xpolicylab_arxiv_2608_09892.md`](../papers/xpolicylab_arxiv_2608_09892.md)
- **代码：** <https://github.com/XPolicyLab/XPolicyLab> — [`sources/repos/xpolicylab.md`](../repos/xpolicylab.md)
- **机构：** MMLab@HKU & THU；项目牵头 Tianxing Chen
- **入库日期：** 2026-08-12
- **一句话说明：** 官方项目站：O(N+M) 集成叙事、42 策略列表、RoboTwin / RoboDojo-sim / RoboDojo-real 榜接线说明与贡献指南。

## 开源核查（步骤 2.5，截至 2026-08-12）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **是** — GitHub `XPolicyLab/XPolicyLab` |
| 训练/评测入口 | **有** — `scripts/create_policy.sh`、`policy/*/eval.sh`、debug/sim 环境 |
| 数据 / 权重 | 各策略自带 checkpoint 下载脚本（HF / ModelScope 优先）；RoboDojo 数据脚本在仓内 |
| 综合判定 | **已开源** |

## 页面要点

- Hero：Connecting N policies to M environments — \(O(NM)\to O(N{+}M)\)。
- 42 policies；同一 adapter 服务 RoboTwin、RoboDojo-sim、RoboDojo-real。
- 贡献：PR 加 `policy/<NAME>/`；官方榜需可复现 checkpoint。
- 受控研究：标准适配 ~2 h；agent skills ~30 min（相对 >5 h 手工）。

## 关联资料

- 论文摘录：[`sources/papers/xpolicylab_arxiv_2608_09892.md`](../papers/xpolicylab_arxiv_2608_09892.md)
- 仓库归档：[`sources/repos/xpolicylab.md`](../repos/xpolicylab.md)
- Wiki：[`wiki/entities/paper-xpolicylab.md`](../../wiki/entities/paper-xpolicylab.md)、[`wiki/entities/xpolicylab.md`](../../wiki/entities/xpolicylab.md)
