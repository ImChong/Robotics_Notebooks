# World Translation（arXiv:2607.18154）

> 来源归档（ingest）

- **论文：** <https://arxiv.org/abs/2607.18154>（2026-07-20 提交）
- **作者：** Xinchen Yao, Leixin Chang, Hua Chen
- **代码/权重/数据：** 截至 2026-07-22，arXiv 未列官方项目页或公开资产
- **一句话说明：** 从已发生的状态转移反向提取不可观测动力学，再用无配对域翻译把动力学内容从仿真映射到现实。

## 核心摘录

- **问题：** 突发接触等事件前的历史并不包含足够信息，前向历史模型无法恢复隐变量。
- **方法：** 从观测到的 transition 反向抽取动态特征，保持 dynamics content、迁移 domain style。
- **结果：** 在人形、四足和机械臂平台上优于基线，并在 Unitree Go2 上验证策略迁移；摘要未报告统一数值。

**对 wiki 的映射：** [`wiki/entities/paper-world-translation.md`](../../wiki/entities/paper-world-translation.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[Domain Randomization](../../wiki/concepts/domain-randomization.md)。
