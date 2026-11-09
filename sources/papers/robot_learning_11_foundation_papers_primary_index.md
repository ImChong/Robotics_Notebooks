# 11 篇机器人学习论文：原始入口与去重映射（2026-11-09）

> 类型：primary bibliography / ingest index。**11 篇一手论文各有通读归档，具体范围以来源文件的“核对范围”与“通读”字段为准。** 这些记录不代表独立复现实验或通读其全部参考论文。

| 编号 | 官方原始论文 | 本库节点或来源 | 状态 |
|---|---|---|---|
| 1 | [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403)（2510.12403） | [sources/courses/robot_learning_tutorial_arxiv_2510_12403.md](../../sources/courses/robot_learning_tutorial_arxiv_2510_12403.md) | 已有教程来源 |
| 2 | [DAVE-2: End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)（1604.07316） | [wiki/entities/paper-dave2-end-to-end-driving.md](../../wiki/entities/paper-dave2-end-to-end-driving.md) | 原论文正文已通读，本次补充实验与局限 |
| 3 | [PilotNet Saliency](https://arxiv.org/abs/1704.07911)（1704.07911） | [wiki/entities/paper-pilotnet-saliency.md](../../wiki/entities/paper-pilotnet-saliency.md) | 原论文正文已通读，本次补充实验与局限 |
| 4 | [Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702)（1504.00702） | [wiki/entities/paper-deep-visuomotor-policies.md](../../wiki/entities/paper-deep-visuomotor-policies.md) | 原论文正文已通读，本次补充实验与局限 |
| 5 | [ACT / ALOHA](https://arxiv.org/abs/2304.13705)（2304.13705） | [wiki/entities/paper-act.md](../../wiki/entities/paper-act.md) | 复用现存节点 |
| 6 | [Diffusion Policy](https://arxiv.org/abs/2303.04137)（2303.04137） | [wiki/entities/paper-diffusion-policy.md](../../wiki/entities/paper-diffusion-policy.md) | 复用现存节点 |
| 7 | [RT-1](https://arxiv.org/abs/2212.06817)（2212.06817） | [wiki/entities/paper-rt-1.md](../../wiki/entities/paper-rt-1.md) | 复用现存节点 |
| 8 | [VLA Action Tokenization Survey](https://arxiv.org/abs/2507.01925)（2507.01925） | [wiki/entities/paper-vla-action-tokenization-survey.md](../../wiki/entities/paper-vla-action-tokenization-survey.md) | 原论文正文已通读，本次补充实验与局限 |
| 9 | [DAgger / Ross et al.](https://arxiv.org/abs/1011.0686)（1011.0686） | [wiki/entities/paper-ross-dagger.md](../../wiki/entities/paper-ross-dagger.md) | 复用现存节点 |
| 10 | [π₀ / pi0 Flow](https://arxiv.org/abs/2410.24164)（2410.24164） | [wiki/entities/paper-pi0.md](../../wiki/entities/paper-pi0.md) | 复用现存节点 |
| 11 | [LIBERO](https://arxiv.org/abs/2306.03310)（2306.03310） | [wiki/entities/libero-benchmark.md](../../wiki/entities/libero-benchmark.md) | 复用现存节点 |

## 核对与去重说明

- 论文题名、编号与作者以 arXiv 原始页面为准；特别校正 [10] 标题缺失的 `π₀`（pi0）。
- 已有项目：ACT、Diffusion Policy、RT-1、DAgger、π₀、LIBERO；教程已有 `sources/courses/` 归档。
- 新增详情：2016 DAVE-2、2017 PilotNet 解释、2015 Deep Visuomotor Policies、2025 VLA Token Survey。
- 对训练方式与开源性的陈述均以各篇官方论文及可核查官方源码为准；`cbfinn/gps` 属 GPS 重实现，`Psi-Robot/Awesome-VLA-Papers` 只有文献目录。

## 后续校验

- 四篇本次新增论文的原始正文、主要图表与所附附录已核查，来源归档已标注“通读：已完成”。教程、ACT、Diffusion Policy、RT-1 需继续独立核验整个正文/附录；DAgger、π₀、LIBERO 已有先前完整通读归档。
- 需要在仓库工作树或 GitHub Actions 环境完成 `make ci-preflight` / `npm run lint:mermaid` 与报告脚本复核。
