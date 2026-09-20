# ASAP: aligning simulation and real-world physics for learning agile humanoid whole-body skills

> 来源归档（paper / 自由度FreeDof Sim2Real 44 篇参考文献 [33/44]）

- **标题：** ASAP: aligning simulation and real-world physics for learning agile humanoid whole-body skills
- **类型：** paper
- **出处：** RSS 2025
- **章节：** 残差学习（[四条路线梳理](https://mp.weixin.qq.com/s/K_6MibGXWwh9OL9eSZxOMg)）
- **arXiv：** <https://arxiv.org/abs/2502.01143>
- **项目页：** <https://agile.human2humanoid.com/>
- **入库日期：** 2026-09-20
- **开源状态：** 待核实
- **一句话说明：** 真机 rollout 学 delta action，冻结后嵌入仿真微调策略，部署时去掉修正模型。
- **沉淀到 wiki：** [`wiki/entities/paper-hrl-stack-25-asap.md`](../../wiki/entities/paper-hrl-stack-25-asap.md)

## 核心摘录（归纳）

- 文内 52.7% 跟踪误差降低案例；动作层残差完整流程代表。
- Sim 预训练 → 真机配对轨迹 → delta action 模型 → 仿真对齐微调 → 真机无 delta 部署。

## 对 wiki 的映射

- [paper-hrl-stack-25-asap](../../wiki/entities/paper-hrl-stack-25-asap.md)
- [freedof-sim2real-44-papers-technology-map](../../wiki/overview/freedof-sim2real-44-papers-technology-map.md)
- [sim2real-four-routes-identifiability](../../wiki/comparisons/sim2real-four-routes-identifiability.md)
