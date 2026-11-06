# 七项机器人资料：初步核验与待完成工作

入库日期：2026-11-06。**状态：草稿；尚未完成论文通读、去重、实体页和验证，不可作为完整 ingest 合并。** 下述论文信息目前仅根据 arXiv v1 摘要核对，数值为作者报告，不是独立复现结果。

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

## 完成前待办

1. 通读五篇论文全文及相关附录；逐项核实原文版本、方法、实验、限制。
2. 打开项目页，核对代码、数据及权重开放状态；有可运行实现时阅读源码及运行入口。
3. 按项目身份检索并复用已有节点；分别整理 sources 和 wiki，补基于原文的 Mermaid 流程图及适用的源码运行时序图。
4. 在完整检出的仓库运行 `make ci-preflight`，检查 CI；本草稿尚未执行。
