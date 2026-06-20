# wuzy2115/oscar-public

> 来源归档

- **标题：** OSCAR（官方实现）
- **类型：** repo
- **组织：** wuzy2115
- **代码：** <https://github.com/wuzy2115/oscar-public>
- **论文：** <https://arxiv.org/abs/2606.04463>
- **项目页：** <https://wuzy2115.github.io/oscar-project-page/>
- **入库日期：** 2026-06-20
- **一句话说明：** OSCAR **跨具身动作条件视频世界模型** 官方仓库：含数据管线、骨架条件训练与 **Cosmos-Predict2.5-2B** 微调脚本；发布处理数据与 checkpoint，支撑 **RoboArena** 类虚拟策略评估复现。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [OSCAR](../../wiki/entities/paper-oscar.md) | 实体归纳页：骨架条件、数据管线、策略评估协议 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | **2B 级** 动作条件视频 WM；显式骨架条件 vs latent-action / pointmap |
| [Cosmos 3](../../wiki/entities/cosmos-3.md) | 基座为 **Cosmos-Predict2.5-2B**（rectified-flow DiT + WAN 2.1 VAE） |
| [RoboArena](../../wiki/methods/roboarena.md) | 论文在 RoboArena 七策略池上验证虚拟评测与真机排名相关性 |
| [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) | 像素级 WM rollout 作策略评估代理环境 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/oscar_arxiv_2606_04463.md`](../papers/oscar_arxiv_2606_04463.md)
- 沉淀 **[`wiki/entities/paper-oscar.md`](../../wiki/entities/paper-oscar.md)**
