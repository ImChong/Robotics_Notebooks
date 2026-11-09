# 11 篇机器人学习论文：原始入口与去重映射（2026-11-09）

> 类型：primary bibliography / ingest index。**本文件是 11 篇论文的检索映射，不冒充 11 篇全文通读记录。** 新增四篇原始阅读归档以各自的“通读”标记为准。

| 编号 | 官方原始论文 | 本库节点或来源 | 状态 |
|---|---|---|---|
| 1 | [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403)（2510.12403） | [sources/courses/robot_learning_tutorial_arxiv_2510_12403.md](../../sources/courses/robot_learning_tutorial_arxiv_2510_12403.md) | 已有教程来源 |
| 2 | [DAVE-2: End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)（1604.07316） | [wiki/entities/paper-dave2-end-to-end-driving.md](../../wiki/entities/paper-dave2-end-to-end-driving.md) | 新增草稿，部分核对 |
| 3 | [PilotNet Saliency](https://arxiv.org/abs/1704.07911)（1704.07911） | [wiki/entities/paper-pilotnet-saliency.md](../../wiki/entities/paper-pilotnet-saliency.md) | 新增草稿，部分核对 |
| 4 | [Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702)（1504.00702） | [wiki/entities/paper-deep-visuomotor-policies.md](../../wiki/entities/paper-deep-visuomotor-policies.md) | 新增草稿，部分核对 |
| 5 | [ACT / ALOHA](https://arxiv.org/abs/2304.13705)（2304.13705） | [wiki/entities/paper-act.md](../../wiki/entities/paper-act.md) | 复用现存节点 |
| 6 | [Diffusion Policy](https://arxiv.org/abs/2303.04137)（2303.04137） | [wiki/entities/paper-diffusion-policy.md](../../wiki/entities/paper-diffusion-policy.md) | 复用现存节点 |
| 7 | [RT-1](https://arxiv.org/abs/2212.06817)（2212.06817） | [wiki/entities/paper-rt-1.md](../../wiki/entities/paper-rt-1.md) | 复用现存节点 |
| 8 | [VLA Action Tokenization Survey](https://arxiv.org/abs/2507.01925)（2507.01925） | [wiki/entities/paper-vla-action-tokenization-survey.md](../../wiki/entities/paper-vla-action-tokenization-survey.md) | 新增草稿，部分核对 |
| 9 | [DAgger / Ross et al.](https://arxiv.org/abs/1011.0686)（1011.0686） | [wiki/entities/paper-ross-dagger.md](../../wiki/entities/paper-ross-dagger.md) | 复用现存节点 |
| 10 | [π₀ / pi0 Flow](https://arxiv.org/abs/2410.24164)（2410.24164） | [wiki/entities/paper-pi0.md](../../wiki/entities/paper-pi0.md) | 复用现存节点 |
| 11 | [LIBERO](https://arxiv.org/abs/2306.03310)（2306.03310） | [wiki/entities/libero-benchmark.md](../../wiki/entities/libero-benchmark.md) | 复用现存节点 |

## 核对与去重说明

- 论文题名、编号与作者以 arXiv 原始页面为准；特别校正 [10] 标题缺失的 `π₀`（pi0）。
- 已有项目：ACT、Diffusion Policy、RT-1、DAgger、π₀、LIBERO；教程已有 `sources/courses/` 归档。
- 新增详情：2016 DAVE-2、2017 PilotNet 解释、2015 Deep Visuomotor Policies、2025 VLA Token Survey。
- 对训练方式与开源性的陈述均以各篇官方论文及可核查官方源码为准；`cbfinn/gps` 属 GPS 重实现，`Psi-Robot/Awesome-VLA-Papers` 只有文献目录。

## 后续校验

- 补读四篇新论文全部正文、实验图表和附录，核对细节后将其对应 `sources/papers/` 的通读标记改为“已完成”，并更新 wiki 健康度。
- 需要本地完成 `make ci-preflight` / `npm run lint:mermaid`，本会话没有可联网克隆仓库的构建环境。
