# SMPC-to-RL 项目页（pages.rai-inst.com/smpc2rl）

> 来源归档（ingest 配套站点）

- **URL：** <https://pages.rai-inst.com/smpc2rl/>
- **标题：** Learning Loco-Manipulation From SMPC Demonstrations With Sparse Offline-to-Online RL
- **机构：** RAI Institute / 慕尼黑工业大学 / 苏黎世联邦理工
- **论文：** <https://arxiv.org/abs/2608.12063> — 归档见 [`sources/papers/smpc2rl_arxiv_2608_12063.md`](../papers/smpc2rl_arxiv_2608_12063.md)
- **入库日期：** 2026-08-14
- **再核日期：** 2026-08-17
- **一句话说明：** RAI 落地页：SMPC 仿真专家 → 稀疏 offline-to-online RL → Spot/G1 真机全身操作。截至再核日 **无 Code 按钮、无 GitHub**。

## 公开信息要点（截至 2026-08-17）

| 项 | 状态 |
|----|------|
| **Paper / BibTeX** | 已链 arXiv:2608.12063；Under submission |
| **Demo 视频** | 有（Spot 推箱/扶胎/滚胎；G1 推箱） |
| **方法叙事** | 三阶段：SMPC 采数 → 稀疏 FastTD3 → 真机；消融含数据量/质量/多模态/专家比例/撤出阈值/有界 critic |
| **代码按钮** | **无**；文中 judo 是通用 SMPC 工具箱，不是本文官方仓 → [`sources/repos/judo.md`](../repos/judo.md) |
| **结论** | 项目页可用于方法与定性结果；**训练/推理代码未发布** |

## 页面结构速记

1. **为何要稀疏** — 稠密奖励改一次就要再训一轮；带臂 Spot 滚重胎几乎无法手调。
2. **数据从哪来** — SMPC 无训练、分钟级调代价；tiled GPU 约 1M 样本/小时。
3. **训练** — 50% 专家 replay → 10% 成功率后撤出；学成策略比教师更快、更稳。
4. **部署** — 增量动作限加速；冻结 ReLIC 低层保平衡；物体 DR + 非对称 critic。
5. **消融 tab** — 数据量 / 质量 / 多模态 / 专家比例 / 撤出阈值 / 有界 critic。

## 关联资料

- 论文摘录：[`sources/papers/smpc2rl_arxiv_2608_12063.md`](../papers/smpc2rl_arxiv_2608_12063.md)
- 对照仓（非本管线）：[`sources/repos/judo.md`](../repos/judo.md)
- Wiki 实体：[`wiki/entities/paper-smpc2rl-loco-manipulation.md`](../../wiki/entities/paper-smpc2rl-loco-manipulation.md)
- 同实验室对照：[Sumo](../../wiki/methods/sumo.md)（MPC-over-RL，在线规划；本页是 SMPC 离线教书）
