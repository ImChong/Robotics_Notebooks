# judo（RAI 采样 MPC 工具箱）

> 来源归档（ingest 配套仓库）

- **标题：** judo — a hackable sampling-based MPC toolbox
- **类型：** repo
- **组织：** RAI Institute（`rai-opensource` / 历史 org `bdaiinstitute`）
- **代码：** <https://github.com/rai-opensource/judo>（PyPI：`judo-rai`）
- **项目页 / 文档：** <https://pages.rai-inst.com/judo/> · <https://bdaiinstitute.github.io/judo>
- **论文：** Li et al., *Judo: A User-Friendly Open-Source Package for Sampling-Based Model Predictive Control*，[arXiv:2506.17184](https://arxiv.org/abs/2506.17184)
- **许可：** MIT（研究原型 / alpha）
- **星标（截至 2026-08-17）：** ~296
- **入库日期：** 2026-08-17
- **一句话说明：** RAI 开源的采样 MPC 工具箱（Predictive Sampling / CEM / MPPI + 实时 GUI）。**不是** [SMPC-to-RL](../papers/smpc2rl_arxiv_2608_12063.md) 的 tiled 采数 + FastTD3 + ReLIC 官方仓。
- **沉淀到 wiki：** [`wiki/entities/paper-smpc2rl-loco-manipulation.md`](../../wiki/entities/paper-smpc2rl-loco-manipulation.md)（对照仓，非本管线）

## 与 SMPC-to-RL 的边界

[SMPC-to-RL 项目页](https://pages.rai-inst.com/smpc2rl/) 把 judo 写成「采样 MPC 实现参考」。judo 能交互调代价、跑通用采样规划器，但：

| 资产 | judo | SMPC-to-RL 论文管线 |
|------|------|---------------------|
| tiled GPU 专家采数 | 无本文 Algorithm 1 | 论文附录 C |
| 稀疏 FastTD3 / 混合 replay | 无 | 论文 §3.2 |
| 冻结 ReLIC 低层 + Spot/G1 任务 | 有 Spot 相关扩展，**不是**本文五任务配方 | 未开源 |

复现本文时不要把 `pip install judo-rai` 当成官方训练入口。

## 入口速查（README）

| 命令 | 作用 |
|------|------|
| `pip install judo-rai` | 安装（可选 `[dev]`） |
| `judo` | 启动仿真 + 浏览器 GUI |
| `pixi shell` / `pixi run build` | 可复现环境；Spot 任务需编译 `mujoco_extensions` |

## 关联资料

- 论文摘录：[`sources/papers/smpc2rl_arxiv_2608_12063.md`](../papers/smpc2rl_arxiv_2608_12063.md)
- 项目页归档：[`sources/sites/rai-inst-smpc2rl.md`](../sites/rai-inst-smpc2rl.md)
