# 七项机器人资料：登记与节点映射

入库日期：2026-11-06。**状态：已完成。** 下方各项的正式节点见「节点映射」；以下为入库时的 v1 摘要级登记（保留作溯源），**与全文核对后的出入以各 wiki 节点为准**（例如 GO-3 三榜第一只有 RoboTwin 完整成立；Compounding Physical RSI 全文只做仿真、不支持真机；HALO 的 91% 是 6 项单任务均值而多任务真机均值为 70%）。

## HALO

- 论文：[Perceptive Humanoid Motion from Simulation](https://arxiv.org/abs/2611.06908v1)
- 项目页：<https://apple-aiml-research.github.io/ml-halo/>
- 摘要称在一千个程序生成住宅中合成视觉、语言和运动数据，并在六个未知真实环境的 360 次试验中取得 91% 成功率。具体任务、数据时长、部署方法和实验口径需通读全文核对。
- 项目页未完成核验；代码及数据的开放状态待确认。

## VABAR

- 论文：[VABAR: A Variational Automation Benchmark for Agentic Robotics](https://arxiv.org/abs/2611.06881v1)
- 项目页：<https://berkeleyautomation.github.io/vabar-website/>
- 摘要列出六类操作任务、33 个技能，以及 Graph-as-Policy 与 Isaac Lab Arena 的基线评测。任务范围包括工业、医疗和基础设施。
- 项目页未完成核验；源码、数据与可运行入口待确认。

## Cosmos 3 交互式世界仿真

- NVIDIA 博文：[New NVIDIA Cosmos 3 Open Models for Interactive World Simulation](https://huggingface.co/blog/nvidia/cosmos-interactive-world-simulation)
- 博文介绍四个 16B Nano 后训练 checkpoint：Transfer-Auto、Sim-Auto、Sim-Transfer、Sim-Bimanual；采用 OpenMDW 1.1 许可，使用受其条款约束。
- 仓库已有 `wiki/entities/cosmos-3.md`，后续应在既有实体中补充并查验模型卡、实现及入口，避免重复节点。

## AGIBOT GO-3 Preview

- 智元新闻稿：[AGIBOT Unveils GO-3 Preview](https://www.prnewswire.com/apac/news-releases/agibot-unveils-go-3-preview-takes-top-spot-on-three-embodied-ai-benchmarks-302927277.html)
- 榜单名次属于厂商声明，需核对榜单、时间和评测口径；模型发布及许可状态待查。

## Compounding Physical RSI

- 论文：[Compounding Physical RSI: Turning Execution Failure into Inherited Capability](https://arxiv.org/abs/2611.06654v1)
- 摘要提出能力缺口评估、离线强化学习修补和权重空间候选筛选；报告 RoboDojo-Sim 上的结果。是否支持真机部署需核对全文。

## A³-VLA

- 论文：[Automatic Perception-Guided Attention Alignment for Robot Manipulation](https://arxiv.org/abs/2611.06447v1)
- 摘要介绍基于交互视觉变化与离线分割生成候选区域，结合注意力分配和对比学习目标；各项实验的基准、绝对数值和适用范围待核对。

## Soccerbot

- 论文：[Playing Soccer Like Humans Do: Learning Versatile and Agile Humanoid Soccer via Latent Motion Priors](https://arxiv.org/abs/2611.06564v1)
- 摘要称通过人类足球动作捕捉构建潜在运动先验，统一移动、带球、停球和踢球，并零样本迁移至 Unitree G1。训练细节与真机试验条件待核对。

## 节点映射（全文阅读与核对后）

| 项 | 节点 | 说明 |
|----|------|------|
| HALO | [paper-halo-perceptive-humanoid-motion-sim](../../wiki/entities/paper-halo-perceptive-humanoid-motion-sim.md) | 20 页含附录读完；代码已公开（Apple 自定义许可），权重与大部分数据不发布 |
| VABAR | [paper-vabar-variational-automation-benchmark](../../wiki/entities/paper-vabar-variational-automation-benchmark.md) | 项目页写 Nov 9 公开仓库，当前不可达；时序图改画相关公开仓库 RoboSimStudio 的评测通路 |
| Cosmos 3 交互式世界仿真 | [paper-cosmos-interactive-world-simulation](../../wiki/entities/paper-cosmos-interactive-world-simulation.md)（既有节点，已补全） | 四个 checkpoint 逐个核对模型卡；更正旧页「三类」与「仅 AgiBot」的说法 |
| AGIBOT GO-3 Preview | [agibot-go-3-preview](../../wiki/entities/agibot-go-3-preview.md) | 新闻稿三榜核对；论文 / 代码 / 权重均未找到 |
| Compounding Physical RSI | [paper-compounding-physical-rsi](../../wiki/entities/paper-compounding-physical-rsi.md) | 仅仿真、无代码；与库内 `physical-rsi` 页的榜单数字出处不同，未合并比较 |
| A³-VLA | [paper-a3-vla-attention-alignment](../../wiki/entities/paper-a3-vla-attention-alignment.md) | 未找到公开代码 |
| Soccerbot | [paper-soccerbot-latent-motion-priors](../../wiki/entities/paper-soccerbot-latent-motion-priors.md) | 训练代码与数据未发布；仓库只有网页 playground |

## 仍未完成

- 在完整检出的仓库运行 `make ci-preflight`（本环境为浅克隆，未执行）。
- VABAR 仓库公开后（项目页写 Nov 9）复核其目录与入口；GO-3 开源后复核。
