# CAST: Alternating State-Value Targets and Expanded Policy Gradients for Model-Based Reinforcement Learning（arXiv:2609.08853）

> 来源归档（ingest）

- **标题：** CAST: Alternating State-Value Targets and Expanded Policy Gradients for Model-Based Reinforcement Learning
- **短名：** CAST
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.08853>
- **PDF：** <https://arxiv.org/pdf/2609.08853>
- **项目/代码：** <https://pietronoah.github.io/cast/>
- **入库日期：** 2026-09-09
- **索引来源：** [sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md](../blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)
- **一句话说明：** CAST（arXiv:2609.08853）：交替 state-value 目标把规划器行为与策略想象轨迹接起来；DMControl+HumanoidBench 1M 步均值 764±63；Go2 真机倒立迁移。

## 开源状态（步骤 2.5，2026-09-09）

- **结论：** **未开源** — 项目/仓库见 `https://pietronoah.github.io/cast/`。
- **核查：** 项目页无 GitHub / Hugging Face 可运行链。

## 核心摘录（面向 wiki 编译）

- 14 任务 1M 步：均值回报 764±63，高于 BMPC 644±43、BOOM 721±41
- Q→V critic + 扩展 k 步策略梯度
- 仿真训练策略零样本迁移 Unitree Go2 动态倒立

**对 wiki 的映射：** [paper-cast-mbrl](../../wiki/entities/paper-cast-mbrl.md)
