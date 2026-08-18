# AutoPSO: A Metaframework for Automated Particle Swarm Optimization（arXiv:2608.07539）

> 来源归档（ingest）

- **标题：** AutoPSO: A Metaframework for Automated Particle Swarm Optimization
- **缩写 / 框架：** **AutoPSO**
- **类型：** paper / evolutionary-computing / pso / neuroevolution / robotics-control
- **arXiv：** <https://arxiv.org/abs/2608.07539>
- **会议 / 期刊：** IEEE TEVC（accepted）
- **代码：** <https://github.com/EMI-Group/AutoPSO>（归档见 [`sources/repos/autopso.md`](../repos/autopso.md)）
- **作者：** Xinmeng Yu、Jiaxin Gao、Jianguo Zhang、Dongmei Jiang、Ran Cheng
- **机构：** 南方科技大学 / EMI-Group（EvoX）
- **入库日期：** 2026-08-18
- **一句话说明：** 把 PSO 变体设计做成双层搜索：外层搜组件组合，内层实例化求解并反馈；EvoX 种群张量化，覆盖数值基准与神经进化机器人控制。

## 开源状态（步骤 2.5）

- **无独立 `*.github.io` 项目页**；以论文 Code 链与 [EMI-Group/AutoPSO](https://github.com/EMI-Group/AutoPSO) 为准。
- **仓库核查（2026-08-18）：** 含 `src/autopso/`、`examples/pytorch/example_cec2022.py`、`pyproject.toml`；依赖 EvoX + PyTorch。无 SPDX LICENSE 文件。
- **结论：** **已开源、可运行**（CEC2022 示例入口可辨识）。

## 摘录 1：问题

手工 PSO 变体跨任务泛化弱，设计空间大、机制难复用；主流实现 CPU 绑定，规模化贵。

## 摘录 2：方法

外层粒子编码一种候选 PSO：`weight`、两套加速系数 `k1–k4`、四类 exemplar `type1–type4`、子群比例 `percent`。内层广义 PSO 把种群切成两策略组，从九候选池选 exemplar。默认外层种群 100、内层 100×800 iter；CEC2022 10D/20D 墙钟预算 60s/120s。

## 摘录 3：数字读法

对六种经典 PSO 在 CEC2022 等墙钟设定下报告更强变体；另含神经进化机器人控制。复现必须 GPU（README：RTX 3090）；CPU 同等 120s 外层迭代会少很多。

**对 wiki 的映射：** [`wiki/entities/paper-autopso.md`](../../wiki/entities/paper-autopso.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（可运行 CEC 示例）
